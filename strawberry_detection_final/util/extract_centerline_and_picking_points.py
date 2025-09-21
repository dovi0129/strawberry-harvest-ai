import numpy as np
import cv2
from skimage.draw import line as bresenham_line
from typing import List

def refine_triangle_vertices(mask: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    refined_pts = []
    for i in range(3):
        p0 = triangle[i]
        p1 = triangle[(i + 1) % 3]
        p2 = triangle[(i + 2) % 3]
        midpoint = ((p1 + p2) / 2).astype(int)
        rr, cc = bresenham_line(p0[1], p0[0], midpoint[1], midpoint[0])
        for y, x in zip(rr, cc):
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
                refined_pts.append(np.array([x, y]))
                break
        else:
            refined_pts.append(p0)
    return np.array(refined_pts, dtype=np.int32)

def extract_centerline_and_picking_points(mask: np.ndarray) -> tuple:
    """
    주어진 딸기 인스턴스 마스크로부터 중심축과 수확지점 후보 두 점을 계산합니다.

    Returns:
        tip (np.ndarray): 중심축 하단 점
        midpoint (np.ndarray): 중심축 상단 점
        picking_pts (list[np.ndarray]): 수확 지점 후보 두 점
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours[0]) < 3:
        return None, None, []

    ret = cv2.minEnclosingTriangle(contours[0])
    if ret is None:
        return None, None, []

    _, triangle = ret
    triangle = np.squeeze(triangle).astype(np.int32)
    if triangle.shape != (3, 2):
        return None, None, []

    refined_pts = refine_triangle_vertices(mask, triangle)
    if refined_pts.shape != (3, 2):
        return None, None, []

    sorted_by_y = sorted(refined_pts, key=lambda p: p[1], reverse=True)
    tip = sorted_by_y[0]
    picking_pts = [sorted_by_y[1], sorted_by_y[2]]
    midpoint = ((picking_pts[0] + picking_pts[1]) / 2).astype(int)

    return tip, midpoint, picking_pts

def extract_and_draw_centerline(image: np.ndarray, instance_mask: np.ndarray, ripe_ids: List[int]) -> np.ndarray:
    """
    중심축 및 수확 지점을 찾아 이미지 위에 그립니다.

    Args:
        image (np.ndarray): 원본 이미지
        instance_mask (np.ndarray): 인스턴스 마스크 (배경 0, 인스턴스별 정수 ID)
        ripe_ids (list[int]): 숙성된 딸기 인스턴스 ID 리스트

    Returns:
        np.ndarray: 시각화된 이미지
    """
    OUTER_TRIANGLE_COLOR = (220, 220, 220)
    REFINED_TRIANGLE_COLOR = (200, 0, 0)
    MIDLINE_COLOR = (220, 220, 220)
    CENTERLINE_COLOR = (255, 255, 0)
    PICKING_POINTS_COLOR = (255, 255, 0)
    CIRCLE_RADIUS = 3

    output = image.copy()

    for inst_id in ripe_ids:
        mask = (instance_mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours[0]) < 3:
            continue

        ret = cv2.minEnclosingTriangle(contours[0])
        if ret is None:
            continue
        _, triangle = ret
        triangle = np.squeeze(triangle).astype(np.int32)
        if triangle.shape != (3, 2):
            continue

        # 외접 삼각형
        cv2.polylines(output, [triangle.reshape((-1, 1, 2))], isClosed=True,
                      color=OUTER_TRIANGLE_COLOR, thickness=1)

        # 중선
        for i in range(3):
            p0 = triangle[i]
            p1 = triangle[(i + 1) % 3]
            p2 = triangle[(i + 2) % 3]
            midpoint = ((p1 + p2) / 2).astype(int)
            cv2.line(output, tuple(p0), tuple(midpoint), MIDLINE_COLOR, 1, lineType=cv2.LINE_AA)

        # 리파인 삼각형
        refined_pts = refine_triangle_vertices(mask, triangle)
        if refined_pts.shape != (3, 2):
            continue
        cv2.polylines(output, [refined_pts.reshape((-1, 1, 2))], isClosed=True,
                      color=REFINED_TRIANGLE_COLOR, thickness=2)

        # 중심축 및 수확점
        sorted_by_y = sorted(refined_pts, key=lambda p: p[1], reverse=True)
        tip = sorted_by_y[0]
        pick1 = sorted_by_y[1]
        pick2 = sorted_by_y[2]
        midpoint = ((pick1 + pick2) / 2).astype(int)

        cv2.line(output, tuple(tip), tuple(midpoint), CENTERLINE_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.circle(output, tuple(pick1), CIRCLE_RADIUS, PICKING_POINTS_COLOR, -1)
        cv2.circle(output, tuple(pick2), CIRCLE_RADIUS, PICKING_POINTS_COLOR, -1)
        image = extract_and_draw_centerline(image, instance_mask, ripe_ids)

    return output
