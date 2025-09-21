# mid_to_move_jetson.py
# Jetson: detection에서 mid(JSON, m) 수신 → 카메라→베이스 변환 → (접근+5cm) → 타겟 → 스캔포즈 복귀
# - 중복/근접 필터: 픽셀 간격 + 3D 비등방 거리(XY=1, Z=0.3), 임계 1.5cm
# - 프로토콜: BUSY / SKIP_DUP / ACCEPT / ACK_DONE
# - angle은 무시(나중에 OpenMANIPULATOR-X 연동 시 사용)

import socket, json, math, numpy as np, time
from neuromeka import IndyDCP2
from comms.gripper_comm import GripperController # ### 수정: 그리퍼 컨트롤러 임포트 ###

# ───────── 통신 ─────────
HOST = "127.0.0.1"
PORT = 9999

# ───────── 로봇 연결 ─────────
ROBOT_IP   = "192.168.0.18"
ROBOT_NAME = "NRMK-Indy7"
ROBOT_PORT = 6066

# ### 수정: 그리퍼 연결 설정 추가 ###
# ───────── 그리퍼 연결 ─────────
GRIPPER_DEVICE = "/dev/ttyUSB0"
GRIPPER_ID     = 15
# ###############################

# ───────── 스캔(대기) 포즈: m/deg ─────────
SCAN_POSE_M_DEG = [
    0.38969744067737605, -0.18662309719899756, 0.6078173856779798,
   82.31416053844744,   113.06865717101573,   82.89537837567426
]

# ───────── 카메라 → EE (회전 I, 번역만, m) ─────────
T_ee_cam = np.eye(4)
T_ee_cam[:3, 3] = np.array([0.10, 0.00, 0.127])

# ───────── 모션 옵션 ─────────
KEEP_ORI     = True
APPROACH_UP  = 0.050
LIFT_Z_BASE  = 0.200
BACKOFF_EE_Z = 0.200

# ───────── 중복/근접 필터 ─────────
PROCESSED_MIN_SEP = 0.015
PIXEL_MIN_SEP     = 20
W_DIST            = (1.0, 1.0, 0.3)

# ---------- 유틸 (변경 없음) ----------
def rpy_deg_to_R(rpy_deg):
    rx, ry, rz = [math.radians(a) for a in rpy_deg]
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[ math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def R_to_rpy_deg(R):
    sy = -R[2,0]; cy = math.sqrt(max(0.0, 1 - sy*sy))
    if cy > 1e-6:
        rx = math.atan2(R[2,1], R[2,2]); ry = math.asin(sy); rz = math.atan2(R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1]); ry = math.asin(sy); rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]

def pose_to_T(pose_m_deg):
    x,y,z, rx,ry,rz = pose_m_deg
    T = np.eye(4); T[:3,:3] = rpy_deg_to_R([rx,ry,rz]); T[:3,3] = [x,y,z]
    return T

def T_to_pose(T):
    x,y,z = T[:3,3].tolist()
    rx,ry,rz = R_to_rpy_deg(T[:3,:3])
    return [x,y,z, rx,ry,rz]

def h(p3):
    return np.array([p3[0], p3[1], p3[2], 1.0], dtype=float)

def mdeg_to_mmrad(pose_m_deg):
    x,y,z, rx,ry,rz = pose_m_deg
    return [x*1000.0, y*1000.0, z*1000.0, math.radians(rx), math.radians(ry), math.radians(rz)]

def wait_idle(robot, timeout=30.0, poll=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if hasattr(robot, "is_idle") and robot.is_idle():
            return True
        time.sleep(poll)
    return False

# ---------- 변환/경로 (변경 없음) ----------
def plan_from_camera_point(p_cam_m, curr_pose_base_m_deg=SCAN_POSE_M_DEG):
    T_base_ee  = pose_to_T(curr_pose_base_m_deg)
    T_base_cam = T_base_ee @ T_ee_cam
    p_base     = (T_base_cam @ h(p_cam_m))[:3]

    if KEEP_ORI:
        R_target = T_base_ee[:3,:3]
    else:
        ee_pos = T_base_ee[:3,3]
        z_axis = p_base - ee_pos; z_axis /= (np.linalg.norm(z_axis)+1e-9)
        t = np.array([0,0,1.0]) if abs(np.dot(z_axis,[0,0,1.0]))<=0.98 else np.array([0,1.0,0])
        x_axis = np.cross(t, z_axis); x_axis /= (np.linalg.norm(x_axis)+1e-9)
        y_axis = np.cross(z_axis, x_axis); y_axis /= (np.linalg.norm(y_axis)+1e-9)
        R_target = np.column_stack([x_axis, y_axis, z_axis])

    ee_plus_z_in_base = R_target[:, 2]
    p_base_backed     = p_base - ee_plus_z_in_base * BACKOFF_EE_Z
    p_base_final      = p_base_backed.copy()
    p_base_final[2]  += LIFT_Z_BASE

    T_target   = np.eye(4); T_target[:3,:3] = R_target; T_target[:3,3] = p_base_final
    T_approach = T_target.copy(); T_approach[2,3] += APPROACH_UP

    approach_pose_m_deg = T_to_pose(T_approach)
    target_pose_m_deg   = T_to_pose(T_target)
    return approach_pose_m_deg, target_pose_m_deg, p_base

# ---------- 중복/근접 판정 (변경 없음) ----------
def is_far_pixel(uv, processed_uv, min_sep=PIXEL_MIN_SEP):
    if uv is None:
        return True
    for puv in processed_uv:
        du = uv[0] - puv[0]; dv = uv[1] - puv[1]
        if (du*du + dv*dv) ** 0.5 < min_sep:
            return False
    return True

def is_far_3d_weighted(p_cam, processed_cam, min_sep=PROCESSED_MIN_SEP, w=W_DIST):
    wx, wy, wz = w
    for q in processed_cam:
        d = ((wx*(p_cam[0]-q[0]))**2 +
             (wy*(p_cam[1]-q[1]))**2 +
             (wz*(p_cam[2]-q[2]))**2) ** 0.5
        if d < min_sep:
            return False
    return True

# ---------- 메인 ----------
def main():
    robot = None
    gripper = None
    # ### 수정: 로봇 및 그리퍼 연결/해제를 위한 try...finally 추가 ###
    try:
        # 로봇 연결
        robot = IndyDCP2(ROBOT_IP, ROBOT_NAME, str(ROBOT_PORT))
        robot.connect()
        time.sleep(0.1)
        print("[ROBOT] Connected.")

        # 그리퍼 연결
        gripper = GripperController(device_name=GRIPPER_DEVICE, motor_id=GRIPPER_ID)
        print("[GRIPPER] Connected.")

        # 시작 시 스캔포즈 및 그리퍼 열기
        robot.task_move_to(mdeg_to_mmrad(SCAN_POSE_M_DEG))
        wait_idle(robot)
        gripper.open() # 시작할 때 그리퍼 열어두기
        print("[ROBOT] Moved to scan pose. Gripper is open.")

        processed_cam = []
        processed_uv  = []
        busy = False

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"[SERVER] Listening on {HOST}:{PORT} ...")
            conn, addr = s.accept()
            print(f"[SERVER] Connected by {addr}")
            conn.sendall(b"[SERVER] Connection established\n")

            with conn:
                buffer = ""
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            print("[SERVER] JSON decode error:", line[:120])
                            continue

                        if "mid" in msg and all(k in msg["mid"] for k in ("x","y","z")):
                            p_cam = np.array([float(msg["mid"]["x"]),
                                              float(msg["mid"]["y"]),
                                              float(msg["mid"]["z"])], dtype=float)
                        else:
                            conn.sendall(b"SKIP_DUP\n")
                            continue

                        uv = None
                        if "mid_uv" in msg and "u" in msg["mid_uv"] and "v" in msg["mid_uv"]:
                            uv = (int(msg["mid_uv"]["u"]), int(msg["mid_uv"]["v"]))

                        if busy:
                            conn.sendall(b"BUSY\n")
                            continue

                        if (not is_far_pixel(uv, processed_uv)) or (not is_far_3d_weighted(p_cam, processed_cam)):
                            conn.sendall(b"SKIP_DUP\n")
                            continue

                        conn.sendall(b"ACCEPT\n")

                        approach_mdeg, target_mdeg, p_base = plan_from_camera_point(p_cam)
                        print("[SERVER] p_cam(m):", p_cam, " p_base(m):", p_base)
                        busy = True
                        try:
                            # 1. 접근 위치로 이동
                            robot.task_move_to(mdeg_to_mmrad(approach_mdeg)); wait_idle(robot)
                            # 2. 최종 목표 위치로 이동
                            robot.task_move_to(mdeg_to_mmrad(target_mdeg));   wait_idle(robot)

                            # ### 수정: 그리퍼 작동 로직 ###
                            print("[GRIPPER] Closing gripper (picking)...")
                            gripper.close()
                            time.sleep(1.0) # 파지 대기

                            print("[GRIPPER] Opening gripper (placing)...")
                            gripper.open()
                            time.sleep(1.0) # 놓기 대기
                            # ###########################

                            # 3. 스캔 위치로 복귀
                            robot.task_move_to(mdeg_to_mmrad(SCAN_POSE_M_DEG)); wait_idle(robot)

                            processed_cam.append(p_cam.copy())
                            if uv is not None:
                                processed_uv.append(uv)

                            conn.sendall(b"ACK_DONE\n")
                            print("[SERVER] Cycle done. Back to scan pose.")
                        finally:
                            busy = False
    
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # ### 수정: 로봇 및 그리퍼 연결 해제 ###
        if robot and robot.is_connected():
            robot.disconnect()
            print("[ROBOT] Disconnected.")
        if gripper and gripper.connected:
            gripper.disconnect()
            print("[GRIPPER] Disconnected.")
        # ######################################

if __name__ == "__main__":
    main()