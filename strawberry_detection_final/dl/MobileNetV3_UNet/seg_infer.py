import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from .model import get_model
from . import config


def load_segmentation_model(weight_path=None):
    """
    segmentation 모델 로드 및 준비
    """
    model = get_model()
    device = config.DEVICE
    weight_path = weight_path or os.path.join('models', 'MobileNetV3_UNet', 'checkpoints', 'best_model.pth')
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_segmentation_on_crop(crop_image: np.ndarray, model, device=None):
    """
    crop_image: np.ndarray (H, W, 3) RGB 이미지
    model: segmentation 모델 (UNet)
    return: 이진 마스크 (np.ndarray, shape = (IMG_SIZE, IMG_SIZE), 값: 0 또는 1)
    """
    if crop_image is None or crop_image.ndim != 3:
        raise ValueError("Invalid crop image provided.")

    device = device or config.DEVICE
    img_resized = cv2.resize(crop_image, (config.IMG_SIZE, config.IMG_SIZE))
    tensor = transforms.ToTensor()(Image.fromarray(img_resized)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(tensor)[0, 0].cpu().numpy()
        binary_mask = (pred_mask > config.THRESH).astype(np.uint8)

    return binary_mask


def overlay_mask(crop_image: np.ndarray, mask: np.ndarray, alpha=0.5):
    """
    입력 crop 이미지와 mask를 시각화하여 반환
    mask: 0 또는 1의 np.ndarray (크기: crop_image 와 동일하거나 resize됨)
    """
    if crop_image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (crop_image.shape[1], crop_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_color = np.zeros_like(crop_image, dtype=np.uint8)
    mask_color[mask == 1] = [0, 0, 255]
    vis = cv2.addWeighted(crop_image, 1 - alpha, mask_color, alpha, 0)
    return vis
