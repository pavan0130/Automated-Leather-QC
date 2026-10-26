import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

class MultiScaleExtractor(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.features = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
            
        self.backbone.layer1.register_forward_hook(get_activation("layer1"))
        self.backbone.layer2.register_forward_hook(get_activation("layer2"))
        self.backbone.layer3.register_forward_hook(get_activation("layer3"))
        
    def forward(self, x):
        self.features.clear()
        _ = self.backbone(x)
        return self.features

def compute_anomaly_map(student_features, teacher_features, out_size=(512, 512)):
    anomaly_map = 0
    for layer in student_features.keys():
        s_feat = F.normalize(student_features[layer], p=2, dim=1)
        t_feat = F.normalize(teacher_features[layer], p=2, dim=1)
        distance = 1.0 - F.cosine_similarity(s_feat, t_feat, dim=1).unsqueeze(1)
        distance = F.interpolate(distance, size=out_size, mode="bilinear", align_corners=False)
        anomaly_map += distance
    return anomaly_map / len(student_features)

def apply_border_mask(anomaly_map: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return anomaly_map
    masked = anomaly_map.copy()
    h, w = masked.shape[:2]
    m_h = min(margin, h // 2)
    m_w = min(margin, w // 2)
    
    masked[:m_h, :] = 0.0
    masked[-m_h:, :] = 0.0
    masked[:, :m_w] = 0.0
    masked[:, -m_w:] = 0.0
    return masked

def postprocess_map(raw_map_tensor, effective_threshold: float, border_margin: int = 32, sigma: float = 4.0):
    map_np = raw_map_tensor.squeeze().cpu().numpy()
    smoothed = cv2.GaussianBlur(map_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smoothed_masked = apply_border_mask(smoothed, border_margin)
    
    binary_mask = (smoothed_masked > effective_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return smoothed_masked, clean_mask

def map_sensitivity_to_threshold(sensitivity_slider: float, base_threshold: float) -> float:
    # 5.0 = 1.0x Base Threshold
    # 10.0 = 0.75x Base Threshold (Prevents full-mask threshold collapse)
    # 1.0 = 1.25x Base Threshold
    multiplier = 1.0 - (sensitivity_slider - 5.0) * 0.05
    multiplier = float(np.clip(multiplier, 0.75, 1.25))
    return max(base_threshold * multiplier, 1e-4)

def get_transforms(cfg, is_train: bool = False):
    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
