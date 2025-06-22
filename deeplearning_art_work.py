# ---------------------------------------------------
# 3개 클래스 뇌종양 분류 및 Grad-CAM 시각화 최종 코드
# ---------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
from tqdm import tqdm

from datetime import datetime
import wandb # [WANDB] wandb 임포트
import kagglehub

from attention_model import AttentionCNN,ResNet18_CBAM,ResNet34_CBAM
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 1. 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")






# 4. 학습 및 평가 함수 (wandb 로깅 기능 추가)
def train_model(model:torch.nn.Module, train_loader:DataLoader, val_loader:DataLoader, config,model_name, patience=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=1e-4)
    model.to(device)

    # [WANDB] 모델의 그래디언트와 파라미터 추적 시작
    wandb.watch(model, criterion, log="all", log_freq=10)

    # --- [EARLY STOPPING] 조기 종료를 위한 변수 초기화 ---
    best_val_loss = float('inf') # 가장 낮은 검증 손실을 기록하기 위한 변수 (무한대로 초기화)
    patience_counter = 0         # 성능 개선이 없는 에포크를 세는 카운터
    # ----------------------------------------------------

    best_val_accuracy = 0.0
    
    # patience=3: 3 에포크 동안 val_loss 개선이 없으면 학습률을 0.5배로 줄임
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # --- [TQDM NESTED] 외부 프로그레스 바: 전체 에포크 용 ---
    epoch_pbar = tqdm(range(config.epochs), desc="Total Epochs")
    
    for epoch in epoch_pbar:
        # --- [TQDM NESTED] 내부 프로그레스 바: 학습 배치 용 ---
        model.train()
        train_loss = 0.0
        # leave=False: 루프 완료 후 프로그레스 바를 남기지 않음
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            train_loss += current_loss
            
            # 내부 프로그레스 바에 현재 배치 손실 표시
            train_pbar.set_postfix(loss=f"{current_loss:.4f}")
        
        train_loss /= len(train_loader)
        
        # --- [TQDM NESTED] 내부 프로그레스 바: 검증 배치 용 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        scheduler.step(val_loss) # val_loss를 기준으로 학습률 조절
        
        # [WANDB] wandb 로깅 (동일)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy})
        
        # --- [TQDM NESTED] 외부 프로그레스 바에 현재까지의 최고 성능 표시 ---
        epoch_pbar.set_postfix(Best_Val_Loss=f"{best_val_loss:.4f}", Current_Acc=f"{accuracy:.2f}%")

        # 조기 종료 및 최고 모델 저장 로직 (동일)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)
            patience_counter = 0
            best_val_accuracy = accuracy
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n!!! {patience} 에포크 동안 검증 손실이 개선되지 않아 조기 종료합니다. !!!")
                epoch_pbar.close() # 프로그레스 바를 닫고 종료
                break
    
    # 학습 루프가 끝난 후, 최종 결과 한 번 더 출력
    if epoch_pbar.n < epoch_pbar.total -1: # 조기 종료된 경우
        pass # 이미 메시지 출력됨
    else: # 정상 종료된 경우
        print("\n학습이 성공적으로 완료되었습니다.")
        
    print(f"최종 저장된 모델의 검증 손실: {best_val_loss:.4f}")
    print(f"학습 완료. 최고 검증 정확도: {best_val_accuracy:.2f}%")
    
    # [WANDB] 최고 성능 모델을 Artifact로 저장
    best_model_artifact = wandb.Artifact(
        name="best_brain_tumor_model",
        type="model",
        description="가장 높은 검증 정확도를 보인 모델",
        metadata={"best_accuracy": best_val_accuracy}
    )
    best_model_artifact.add_file(model_name)
    wandb.run.log_artifact(best_model_artifact)
    print("최고 성능 모델을 wandb Artifact로 저장했습니다.")
    
    # 마지막으로 최고 성능을 보인 모델의 파라미터를 다시 불러옴
    print(f"최고 성능 모델 '{model_name}'을 불러옵니다.")
    model.load_state_dict(torch.load(model_name))

    return model


# 6. Grad-CAM 및 시각화 함수 정의 (오류 수정된 버전)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    def save_feature_maps(self, module, input, output): self.feature_maps = output.detach()
    def save_gradients(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()
    def __call__(self, x, index=None):
        self.model.eval(); output = self.model(x)
        if index is None: index = torch.argmax(output, dim=1)
        one_hot = torch.zeros_like(output); one_hot.scatter_(1, index.unsqueeze(1), 1)
        self.model.zero_grad(); output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = nn.functional.relu(cam)
        cam_resized = resize(cam.cpu().numpy().squeeze(), (x.shape[2], x.shape[3]))
        cam_resized -= np.min(cam_resized)
        cam_resized /= (np.max(cam_resized) + 1e-8)
        return cam_resized

# ★★★ 3. 시각화 함수 수정: 1채널 이미지를 올바르게 표시 ★★★
def visualize_result(img_tensor, cam, predicted_class, actual_class):
    # 1채널용 역정규화
    inv_normalize = transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5])
    img_for_viz = inv_normalize(img_tensor)
    
    # 텐서를 numpy 배열로 변환. .squeeze()로 채널 차원 제거 -> (H, W) 형태
    img = img_for_viz.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)

    # 히트맵 생성 (컬러)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 원본 흑백 이미지를 3채널로 변환하여 히트맵과 합성
    img_rgb = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2RGB)
    superimposed_img = np.uint8(heatmap * 0.4 + img_rgb * 0.6)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    title = f"Actual: {actual_class} | Predicted: {predicted_class}"
    fig.suptitle(title, fontsize=16)

    # 원본 이미지는 cmap='gray'로 표시
    axs[0].imshow(img, cmap='gray'); axs[0].set_title('Original Image (Grayscale)'); axs[0].axis('off')
    axs[1].imshow(heatmap); axs[1].set_title('Grad-CAM Heatmap'); axs[1].axis('off')
    axs[2].imshow(superimposed_img); axs[2].set_title('Superimposed Image'); axs[2].axis('off')
    plt.show()
    
    

# ===================================================================
# 1. 특정 화가만 선택하는 커스텀 데이터셋 클래스 정의
# ===================================================================
class ArtistSubsetDataset(Dataset):
    def __init__(self, data_dir, artists_to_include, transform=None):
        """
        Args:
            data_dir (string): 'images/images' 와 같이 화가 폴더들이 있는 경로.
            artists_to_include (list): 포함할 화가 이름의 리스트.
            transform (callable, optional): 샘플에 적용될 transform.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.artists_to_include = artists_to_include
        
        self.image_paths = []
        self.labels = []
        
        # 선택된 화가들을 기준으로 클래스와 인덱스를 매핑
        self.class_names = sorted(self.artists_to_include)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.classes=self.class_names
        
        print(f"선택된 화가들로 데이터셋을 구성합니다: {self.class_names}")

        # 선택된 화가 폴더만 순회
        for artist_name in self.class_names:
            artist_path = os.path.join(data_dir, artist_name)
            
            if not os.path.isdir(artist_path):
                print(f"경고: '{artist_name}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 해당 화가의 모든 이미지 파일 경로와 레이블을 리스트에 추가
            for file_name in os.listdir(artist_path):
                # .jpg, .png 등 이미지 파일만 추가
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(artist_path, file_name))
                    self.labels.append(self.class_to_idx[artist_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지를 3채널(RGB)로 불러오기 (전이 학습 모델 호환)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로드 에러: {image_path}, 에러: {e}")
            # 에러 발생 시, 임의의 검은 이미지와 레이블 반환 또는 다른 이미지로 대체
            return torch.randn(3, 224, 224), torch.tensor(label)


        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    CURRENT_FOLDER = os.path.abspath(os.path.dirname(__file__))

    kaggle_dir="ikarus777/best-artworks-of-all-time"

    data_dir=kagglehub.dataset_download(kaggle_dir)
    data_dir=os.path.join(data_dir,'images','images')
    print(data_dir)
    
    # 3. 데이터 변환(Transforms) 정의
    
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # 이미지 일부를 확대/크롭
        transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전
        transforms.RandomRotation(degrees=20),    # 회전 각도 늘리기
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 이미지 일부 이동
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 색상 변화 더 주기
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    ARTISTS_TO_INCLUDE = [
        "Vincent_van_Gogh",
        "Claude_Monet",
        "Leonardo_da_Vinci",
        "Rembrandt",
        "Pablo_Picasso"
        # "Edgar Degas",
        # "Salvador Dali",
        # "Paul Cezanne"
    ]

    #4. 데이터 준비
    full_dataset = ArtistSubsetDataset(
        data_dir=data_dir, 
        artists_to_include=ARTISTS_TO_INCLUDE, 
        transform=train_transforms
    )
    class_names = full_dataset.classes
    print(f"발견된 클래스: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms # 검증셋에는 데이터 증강 없는 transform 적용

    print(f"총 데이터: {len(full_dataset)}개, 학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")


    # 6. 전체 로직 실행
    # [WANDB] 실험 초기화 및 설정


    # [WANDB] 하이퍼파라미터를 딕셔너리로 먼저 정의
    config_dict = {
        "learning_rate": 0.0001,
        "epochs": 1000,
        "batch_size": 32,
        "architecture": "ResNet34_CBAM", # 모델 이름 변경
        "dataset": kaggle_dir,
        "image_size": IMG_SIZE,
    }
    
    # [WANDB] ★★★ 동적 실행 이름 생성 ★★★
    run_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_{config_dict['architecture']}_lr{config_dict['learning_rate']}"
    
    wandb.init(
        entity="0311ben-kwangwoon-university",
        project=kaggle_dir.split('/')[-1], # wandb 프로젝트 이름
        
        # ★★★ 생성된 동적 이름을 name 인자로 전달 ★★★
        name=run_name,
        
        # 하이퍼파라미터 설정
        config=config_dict
    )
    # [WANDB] config 객체를 통해 하이퍼파라미터 접근
    config = wandb.config

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=14,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=14,pin_memory=True)

    best_model_dir=os.path.join(CURRENT_FOLDER,run_name+'best_model.pt')

    # 모델 생성 및 학습
    model = ResNet34_CBAM(num_classes=len(class_names))
    model = train_model(model, train_loader, val_loader, config,best_model_dir)

    print("\n--- 검증 데이터셋의 샘플에 대한 결과 시각화 및 wandb 로깅 ---")
    # 시각화할 이미지 인덱스 선택
    indices_to_visualize = random.sample(range(len(val_dataset)), 5) # 5개 랜덤 샘플
    wandb_images = []

    for index in indices_to_visualize:
        sample_img, sample_label_idx = val_dataset[index]
        actual_class_name = class_names[sample_label_idx]
        
        # Grad-CAM 생성 로직 ...
        input_tensor = sample_img.unsqueeze(0).to(device)
        model.eval() # 모델을 평가 모드로
        with torch.no_grad():
            outputs = model(input_tensor); _, predicted_idx_tensor = torch.max(outputs, 1)
            predicted_idx = predicted_idx_tensor.item(); predicted_class_name = class_names[predicted_idx]
        
        # target_layer = model.features[-2]
        target_layer = model.layer4[-1] 
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam(input_tensor, index=predicted_idx_tensor)
        
        # 로깅할 이미지 생성
        # visualize_result 함수를 약간 수정하여 plt 객체를 반환하게 하거나, 여기서 직접 생성
        inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(IMG_MEAN, IMG_STD)], std=[1/s for s in IMG_STD])
        img_for_viz = inv_normalize(sample_img)
        img = img_for_viz.squeeze().permute(1, 2, 0).cpu().numpy(); img = np.clip(img, 0, 1)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET); heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = np.uint8(heatmap * 0.4 + np.uint8(img * 255) * 0.6)
        
        title = f"Idx:{index} | Actual:{actual_class_name} | Pred:{predicted_class_name}"
        
        wandb_images.append(wandb.Image(superimposed_img, caption=title))

    # [WANDB] 준비된 모든 이미지를 한번에 로깅
    wandb.log({"predictions_and_gradcam": wandb_images})
    print("Grad-CAM 결과 이미지를 wandb에 로깅했습니다.")

    # [WANDB] 실험 종료
    wandb.finish()