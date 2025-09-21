import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import torch
import time
from socket_sender import SocketSender
from util.generate_instance_mask import generate_instance_mask
from util.classify_strawberry_maturity import classify_strawberry_maturity
from util.extract_centerline_and_picking_points import extract_centerline_and_picking_points
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 경로 설정 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'dl', 'yolov5n'))
sys.path.insert(0, os.path.join(BASE_DIR, 'dl', 'MobileNetV3_UNet'))

from dl.MobileNetV3_UNet.seg_infer import load_segmentation_model, infer_segmentation_on_crop
from dl.yolov5n.yolov5_infer import YOLOv5nInfer

# -------------------- 모델 및 장치 초기화 --------------------
yolo_model_path = 'dl/yolov5n/best.pt'
seg_model_path = 'dl/MobileNetV3_UNet/checkpoints/best_model.pth'
DEVICE = 'cuda'

print("[INFO] 모델 로딩 중...")
yolo_model = YOLOv5nInfer(model_path=yolo_model_path, device=DEVICE)
seg_model = load_segmentation_model(seg_model_path)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 센서 설정 적용 (start 전에!)
ctx = rs.context()
device = ctx.query_devices()[0]  # 첫 번째 연결된 장치 사용
depth_sensor = device.first_depth_sensor()

preset_map = {
    'default': 1,
    'high_accuracy': 3,
    'high_density': 4,
    'medium_density': 5
}

depth_sensor.set_option(rs.option.visual_preset, preset_map['high_accuracy'])
depth_sensor.set_option(rs.option.laser_power, 240.0)
depth_sensor.set_option(rs.option.exposure, 8500.0)
depth_sensor.set_option(rs.option.gain, 16.0)

# -------------------- 전송 데이터 설정 --------------------
def pixel_to_meter(x, y, depth_mm, fx=615, fy=615, cx=320, cy=240):
    z = depth_mm / 1000.0  # mm → m
    x_m = (x - cx) * z / fx
    y_m = (y - cy) * z / fy
    return x_m, y_m, z

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# -------------------- 센서/필터 설정 --------------------
def configure_sensor(device):
    depth_sensor = device.first_depth_sensor()
    preset_map = {
        'default': 1,
        'high_accuracy': 3,
        'high_density': 4,
        'medium_density': 5
    }
    depth_sensor.set_option(rs.option.visual_preset, preset_map['high_accuracy'])  # 또는 'default'
    depth_sensor.set_option(rs.option.laser_power, 317.0)
    depth_sensor.set_option(rs.option.exposure, 8500.0)
    depth_sensor.set_option(rs.option.gain, 16.0)

# -------------------- Depth 계산 --------------------
def get_mean_valid_depth_in_mask(depth_frame, mask, padding=6):
    """
    인스턴스 마스크 내부에서 padding을 제거한 후 유효한 depth의 평균을 계산
    """
    depth = np.asanyarray(depth_frame.get_data())
    kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    valid_mask = (depth > 0) & np.isfinite(depth)
    masked_depth = depth[(eroded_mask == 1) & valid_mask]
    
    if masked_depth.size > 0:
        print(f'유효 depth값 개수 ={masked_depth.size}개')
        return float(np.mean(masked_depth))
    else:
        return None

def compute_angle(tip, midpoint):
    dx = midpoint[0] - tip[0]
    dy = tip[1] - midpoint[1]
    angle_rad = np.arctan2(dx, dy)
    return np.degrees(angle_rad)

# -------------------- 메인 루프 --------------------
def main():
    frame_idx = 0
    prev_time = time.time()
    start_time = None
    total_frames = 0

    sender = SocketSender(host='127.0.0.1', port=9999)
    sender.connect()

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        preds = yolo_model(image, frame_idx)
        if preds is None or len(preds) == 0:
            continue

        # 객체 탐지 → 분할
        for (*xyxy, conf, cls) in preds:
            x1 = max(int(xyxy[0].item()), 0)
            y1 = max(int(xyxy[1].item()), 0)
            x2 = min(int(xyxy[2].item()), image.shape[1])
            y2 = min(int(xyxy[3].item()), image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mask = infer_segmentation_on_crop(crop_rgb, seg_model, device=DEVICE)
            mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            binary_mask[y1:y2, x1:x2][mask_resized == 1] = 255

        instance_mask = generate_instance_mask(binary_mask)

        instance_centers = []
        for inst_id in np.unique(instance_mask):
            if inst_id == 0:
                continue
            mask = (instance_mask == inst_id)
            ys, xs = np.where(mask)
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            instance_centers.append((inst_id, center_x, center_y))

        instance_centers.sort(key=lambda x: x[1])

        # 딸기 처리
        for inst_id, cx, cy in instance_centers:
            mask = (instance_mask == inst_id)
            maturity = classify_strawberry_maturity(hsv_image, mask)
            if maturity == 'fully_ripe':
                tip, midpoint, picking_pts = extract_centerline_and_picking_points(mask.astype(np.uint8))
                if tip is not None and midpoint is not None and len(picking_pts) == 2:
                    angle = compute_angle(tip, midpoint)
                    depth_value = get_mean_valid_depth_in_mask(depth_frame, mask.astype(np.uint8), padding=6)
                    if depth_value is not None:
                        print(f"[frame {frame_idx}] ripe_id={inst_id} depth={depth_value / 10.0:.1f}cm")

                        # 좌표 변환
                        left_pt, right_pt = picking_pts
                        left_xyz = pixel_to_meter(*left_pt, depth_value)
                        right_xyz = pixel_to_meter(*right_pt, depth_value)

                        # ✅ mid 좌표 추가
                        mid_xyz = (
                            (left_xyz[0] + right_xyz[0]) / 2,
                            (left_xyz[1] + right_xyz[1]) / 2,
                            (left_xyz[2] + right_xyz[2]) / 2
                        )

                        # 메시지 작성
                        message = {
                            "left":  {"x": round(left_xyz[0], 3), "y": round(left_xyz[1], 3), "z": round(left_xyz[2], 3)},
                            "right": {"x": round(right_xyz[0], 3), "y": round(right_xyz[1], 3), "z": round(right_xyz[2], 3)},
                            "mid":   {"x": round(mid_xyz[0], 3),  "y": round(mid_xyz[1], 3),  "z": round(mid_xyz[2], 3)},
                            "angle": round(angle, 2)
                        }
                        sender.send_data(message)
                    else:
                        print(f"[frame {frame_idx}] ripe_id={inst_id} depth=N/A")

                    print(f"[frame {frame_idx}] ripe_id={inst_id} angle={angle:.2f}° tip={tip}, midpoint={midpoint}")
                break

        # FPS 계산
        if start_time is None:
            start_time = time.time()
        frame_idx += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = curr_time
        total_frames += 1
        total_elapsed = curr_time - start_time
        avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
        print(f"[frame {frame_idx}] FPS: {fps:.2f} | AVG: {avg_fps:.2f}")

    sender.close()
    pipeline.stop()

if __name__ == '__main__':
    main()
