import cv2
import numpy as np

def classify_strawberry_maturity(hsv_image, instance_mask,
                                  v_thresh=30, s_thresh=50,
                                  red_low=10, red_high=170,
                                  ripe_ratio=0.2, semi_ratio=0.1):
    """
    HSV 색 공간과 인스턴스 마스크 기반 딸기 숙성도 분류
    :param hsv_image: 전체 HSV 이미지 (3채널 numpy 배열)
    :param instance_mask: 개별 인스턴스 마스크 (2D, 값은 0 또는 1)
    :return: 'fully_ripe', 'semi_ripe', 'unripe', 'unknown'
    """
    h, s, v = cv2.split(hsv_image)
    valid = (v > v_thresh) & (s > s_thresh) & (instance_mask == 1)
    total = valid.sum()
    if total == 0:
        return 'unknown'
    red_mask = valid & ((h < red_low) | (h > red_high))
    red_ratio = red_mask.sum() / total
    if red_ratio >= ripe_ratio:
        return 'fully_ripe'
    elif red_ratio >= semi_ratio:
        return 'semi_ripe'
    else:
        return 'unripe'
