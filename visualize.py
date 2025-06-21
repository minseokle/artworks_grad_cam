import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from skimage.transform import resize
import cv2

from attention_model import *

class GradCAM:
    def __init__(self, model:nn.Module, target_layer:CBAM):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    def save_feature_maps(self, module, input, output): 
        self.feature_maps = output.detach()
    def save_gradients(self, module, grad_in, grad_out): 
        self.gradients = grad_out[0].detach()
       
    def __call__(self, x:torch.Tensor, index=None):
        self.model.eval()
        output:torch.Tensor = self.model(x)
        if index is None: 
            index = torch.argmax(output, dim=1)
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, index.unsqueeze(1), 1)
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = nn.functional.relu(cam)
        cam_resized = resize(cam.cpu().numpy().squeeze(), (x.shape[2], x.shape[3]))
        cam_resized -= np.min(cam_resized)
        cam_resized /= (np.max(cam_resized) + 1e-8)
        return cam_resized
    
def image_to_res(model:nn.Module,image:np.ndarray|torch.Tensor,img_mean:list[float],img_std:list[float],device)->tuple[int , dict[str,np.ndarray]]:
    
    # Grad-CAM 생성 로직 ...
    if isinstance(image,np.ndarray):
        image=torch.from_numpy(image)
    
    input_tensor = image.to(device)
    model.eval() # 모델을 평가 모드로
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx_tensor = torch.max(outputs, 1)
        predicted_idx = predicted_idx_tensor.item()
    
    target_layer = model.features[-2]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(input_tensor, index=predicted_idx_tensor)
    
    img_dict:dict[str,np.ndarray]={}
    # 로깅할 이미지 생성
    # visualize_result 함수를 약간 수정하여 plt 객체를 반환하게 하거나, 여기서 직접 생성
    inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(img_mean, img_std)], std=[1/s for s in img_std])
    img_for_viz:torch.Tensor = inv_normalize(image)
    img = img_for_viz.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = np.uint8(heatmap * 0.4 + np.uint8(img * 255) * 0.6)
    img_dict['original'] = img
    img_dict['heatmap'] = heatmap
    img_dict['suprimposed'] = superimposed_img
    
    return predicted_idx,img_dict
