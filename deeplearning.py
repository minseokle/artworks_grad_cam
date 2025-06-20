# ---------------------------------------------------
# 3개 클래스 뇌종양 분류 및 Grad-CAM 시각화 최종 코드
# ---------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

# 1. 기본 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")


# 2. 어텐션(CBAM) 및 CNN 모델 정의
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x)); max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1); x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__(); self.ca = ChannelAttention(in_planes, ratio); self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x; x = self.sa(x) * x
        return x

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=3): # 클래스 개수를 인자로 받음
        super(AttentionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), CBAM(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), CBAM(64), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes) # ★★★ 핵심: 클래스 개수에 맞게 출력 뉴런 수 변경 ★★★
        )
    def forward(self, x):
        x = self.features(x); x = self.classifier(x)
        return x

# 3. 데이터 변환(Transforms) 정의
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# train_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
# ])

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5), # 좌우 반전
    transforms.RandomRotation(degrees=20),    # 회전 각도 늘리기
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 이미지 일부 이동
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 색상 변화 더 주기
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])


# 4. 데이터 로딩 및 준비
data_dir = '/root/workspace/2/Brain_Cancer raw MRI data/Brain_Cancer' 
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
class_names = full_dataset.classes
print(f"발견된 클래스: {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms # 검증셋에는 데이터 증강 없는 transform 적용

print(f"총 데이터: {len(full_dataset)}개, 학습 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# 5. 학습 및 평가 함수 정의
# 5. 학습 및 평가 함수 정의 (주기적 저장 기능 추가)
def train_model(model, train_loader, val_loader, epochs=100, model_name="best_model.pt"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)
    
    best_val_accuracy = 0.0 # 최고 정확도 기록용

    print("모델 학습을 시작합니다...")
    for epoch in range(epochs):
        model.train()
        # ... (이하 학습 로직은 이전과 동일) ...
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 검증 로직 (이전과 동일)
        model.eval()
        # ... (이하 검증 로직은 이전과 동일) ...
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"에포크 [{epoch+1}/{epochs}], 학습 손실: {running_loss/len(train_loader):.4f}, 검증 정확도: {accuracy:.2f}%")

        # --- 모델 저장 로직 (두 가지 방식 모두 사용) ---

        # 1. 최고 성능 모델 저장 (기존 방식)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), model_name)
            print(f"★★ 최고 정확도 경신({accuracy:.2f}%)! 모델을 '{model_name}'으로 저장합니다. ★★")
            
        # 2. 20 에포크마다 주기적으로 체크포인트 저장 (새로 추가된 기능)
        # (epoch + 1)이 20의 배수이거나, 마지막 에포크일 경우 저장
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f"== 에포크 {epoch+1} 체크포인트를 '{checkpoint_path}'에 저장했습니다. ==")


    print(f"학습 완료. 최고 검증 정확도: {best_val_accuracy:.2f}%")
    return model



# 6. Grad-CAM 및 시각화 함수 정의 (오류 수정된 버전)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer; self.feature_maps = None; self.gradients = None
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

def visualize_result(img_tensor, cam, predicted_class, actual_class):
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    img_for_viz = inv_normalize(img_tensor)
    img = img_for_viz.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = np.uint8(heatmap * 0.4 + np.uint8(img * 255) * 0.6)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    title = f"Actual: {actual_class} | Predicted: {predicted_class}"
    fig.suptitle(title, fontsize=16)

    axs[0].imshow(img); axs[0].set_title('Original Image'); axs[0].axis('off')
    axs[1].imshow(heatmap); axs[1].set_title('Grad-CAM Heatmap'); axs[1].axis('off')
    axs[2].imshow(superimposed_img); axs[2].set_title('Superimposed Image'); axs[2].axis('off')
    plt.show()


# 7. 전체 로직 실행
# 모델 생성
model = AttentionCNN(num_classes=len(class_names))

# 모델 학습
trained_model = train_model(model, train_loader, val_loader, epochs=10000)

# Grad-CAM 시각화를 위한 샘플 이미지 선택
sample_img, sample_label_idx = val_dataset[10] # 검증 데이터셋에서 11번째 이미지 선택
actual_class_name = class_names[sample_label_idx]

# 모델 예측
input_tensor = sample_img.unsqueeze(0).to(device)
trained_model.eval()
with torch.no_grad():
    outputs = trained_model(input_tensor)
    _, predicted_idx_tensor = torch.max(outputs, 1)
    predicted_idx = predicted_idx_tensor.item()
    predicted_class_name = class_names[predicted_idx]

# Grad-CAM 생성 및 시각화
target_layer = trained_model.features[-2] # 마지막 CBAM 블록
grad_cam = GradCAM(trained_model, target_layer)
cam = grad_cam(input_tensor, index=predicted_idx_tensor) # 모델이 예측한 클래스 기준으로 생성

print("\n--- Grad-CAM 시각화 ---")
print(f"선택된 샘플의 실제 클래스: {actual_class_name}")
print(f"모델의 예측 클래스: {predicted_class_name}")

visualize_result(sample_img, cam, predicted_class_name, actual_class_name)