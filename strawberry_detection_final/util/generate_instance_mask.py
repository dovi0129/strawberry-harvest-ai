import cv2
import numpy as np
from skimage.segmentation import watershed

def generate_instance_mask(mask_gray: np.ndarray, morph_kernel_size=(3, 3), dist_thresh_ratio=0.4):
    """
    바이너리 마스크로부터 watershed 기반 인스턴스 마스크 생성
    :param mask_gray: 2D uint8 바이너리 마스크 (255 또는 0)
    :return: 인스턴스 마스크 (int32 배열), 값: 0=배경, 1~N=인스턴스
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    clean_mask = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    _, markers = cv2.connectedComponents(sure_fg)
    instance_mask = watershed(-dist, markers, mask=(clean_mask > 0))
    return instance_mask
