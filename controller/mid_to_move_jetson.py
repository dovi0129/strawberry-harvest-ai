# mid_to_move_jetson.py
# Jetson: detection에서 mid(JSON, m) 수신 → 카메라→베이스 변환 → (접근+5cm) → 타겟 → 스캔포즈 복귀
# - 중복/근접 필터: 픽셀 간격 + 3D 비등방 거리(XY=1, Z=0.3), 임계 1.5cm
# - 배칭: 0.2s 동안 후보를 모아 '가장 왼쪽' 1개만 선택
# - angle은 무시(나중에 OpenMANIPULATOR-X 연동 시 사용)

import socket, json, math, numpy as np, time
from collections import deque
from neuromeka import IndyDCP2

# ───────── 통신 ─────────
HOST = "127.0.0.1"
PORT = 9999

# ───────── 로봇 연결 ─────────
ROBOT_IP   = "192.168.0.18"
ROBOT_NAME = "NRMK-Indy7"
ROBOT_PORT = 6066

# ───────── 카메라 → EE (회전 I, 번역만, m) ─────────
T_ee_cam = np.eye(4)
T_ee_cam[:3, 3] = np.array([0.10, 0.00, 0.127])

# ───────── 모션 옵션 ─────────
KEEP_ORI     = True
APPROACH_UP  = 0.050   # m
LIFT_Z_BASE  = 0.200   # m
BACKOFF_EE_Z = 0.200   # m

# ───────── 중복/근접 필터 ─────────
PROCESSED_MIN_SEP = 0.015         # m (1.5 cm)
PIXEL_MIN_SEP     = 20            # px
W_DIST            = (1.0, 1.0, 0.3)

# ───────── 배칭(왼쪽 선택을 위한 짧은 수집 창) ─────────
CANDIDATE_WINDOW_SEC = 0.20   # 200 ms
MAX_QUEUE = 256

# ---------- 유틸 ----------
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

def mmrad_to_mdeg(pose_mm_rad):
    x_mm, y_mm, z_mm, rx, ry, rz = pose_mm_rad
    return [x_mm/1000.0, y_mm/1000.0, z_mm/1000.0,
            math.degrees(rx), math.degrees(ry), math.degrees(rz)]

def wait_idle(robot, timeout=30.0, poll=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if hasattr(robot, "is_idle") and robot.is_idle():
            return True
        time.sleep(poll)
    return False

def deg_list_to_rad(lst_deg):
    return [math.radians(v) for v in lst_deg]

# ---------- 변환/경로 ----------
def plan_from_camera_point(p_cam_m, curr_pose_base_m_deg):
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

# ---------- 중복/근접 판정 ----------
def is_far_pixel(uv, processed_uv, min_sep=PIXEL_MIN_SEP):
    if uv is None:
        return True
    for puv in processed_uv:
        du = uv[0]-puv[0]; dv = uv[1]-puv[1]
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

# ---------- 초기 joint 시퀀스 실행 (네가 준 것) ----------
INIT_JOINT_SEQ_DEG = [8, -2, -15, 0, 105, -35]


def run_initial_joint_sequence(robot):
    print("[ROBOT] running initial joint_move_by sequence (deg → rad)")
    for i, step_deg in enumerate(INIT_JOINT_SEQ_DEG, 1):
        robot.joint_move_by(deg_list_to_rad(step_deg))
        wait_idle(robot)
        print(f"  - step {i}/{len(INIT_JOINT_SEQ_DEG)} done")
    # 완료 후 현재 태스크 포즈를 읽어 스캔포즈로 사용
    pose_mm_rad = robot.get_task_pos()  # [mm, mm, mm, rad, rad, rad]
    scan_pose_m_deg = mmrad_to_mdeg(pose_mm_rad)
    print("[ROBOT] initial sequence finished → SCAN_POSE_M_DEG set to:", scan_pose_m_deg)
    return scan_pose_m_deg
ppend(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()

# ---------- 서버 메인 ----------
def main():
    # 로봇 연결 & 초기 joint 시퀀스로 스캔 포즈 설정
    robot = IndyDCP2(ROBOT_IP, ROBOT_NAME, str(ROBOT_PORT))
    robot.connect(); time.sleep(0.1)

    # (1) 네가 준 joint_move_by 시퀀스로 초기 위치 맞추ppend(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()
기
    SCAN_POSE_M_DEG = run_initial_joint_sequence(robot)

    # (2) 안전상 한번 해당 스캔 포즈로 task_move_to (정합 확인)
    robot.task_move_to(mdeg_to_mmrad(SCAN_POSE_M_DEG)); wait_idle(robot)
    print("[ROBOT] moved to scan pose (confirmed)")ppend(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()


    processed_cam = []      # np.array([x,y,z]) m
    processed_uv  = []      # (u,v)
    busy = False
ppend(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()

    candidates = deque(maxlen=MAX_QUEUE)
    first_ts = Noneppend(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()


    # 내부 헬퍼: 메시지를 후보로 푸시
    def _push_candidate(msg):
        nonlocal first_ts, candidates
        p_cam = None
        uv    = None
        if "mid" in msg and all(k in msg["mid"] for k in ("x","y","z")):
            p_cam = np.array([float(msg["mid"]["x"]),
                              float(msg["mid"]["y"]),
                              float(msg["mid"]["z"])], dtype=float)
            if "mid_uv" in msg and "u" in msg["mid_uv"] and "v" in msg["mid_uv"]:
                uv = (int(msg["mid_uv"]["u"]), int(msg["mid_uv"]["v"]))
        elif "left" in msg and "right" in msg:
            lx,ly,lz = float(msg["left"]["x"]),  float(msg["left"]["y"]),  float(msg["left"]["z"])
            rx,ry,rz = float(msg["right"]["x"]), float(msg["right"]["y"]), float(msg["right"]["z"])
            p_cam = np.array([(lx+rx)/2.0, (ly+ry)/2.0, (lz+rz)/2.0], dtype=float)
        else:
            return

        candidates.append({"ts": time.time(), "p_cam": p_cam, "uv": uv})
        if first_ts is None:
            first_ts = candidates[-1]["ts"]

    def pick_leftmost_and_clear():
        nonlocal candidates
        if not candidates:
            return None
        def key_fn(c):
            uv = c.get("uv")
            if uv is not None:
                return (0, uv[0])  # uv.u 작을수록 왼쪽
            return (1, c["p_cam"][0])  # cam x 작을수록 왼쪽 (fallback)
        viable = []
        for c in list(candidates):
            if not is_far_pixel(c.get("uv"), processed_uv):        # 픽셀 근접 제외
                continue
            if not is_far_3d_weighted(c["p_cam"], processed_cam):  # 3D 근접 제외
                continue
            viable.append(c)
        target = min(viable, key=key_fn) if viable else None
        candidates.clear()
        return target

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT)); s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT} ...")
        conn, addr = s.accept()
        print(f"[SERVER] Connected by {addr}")
        try:
            conn.sendall(b"[SERVER] Connection established\n")
        except: pass

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
                    print("[SERVER] JSON decode error(line)", line[:120])
                    continue
                _push_candidate(msg)

            try:
                msg = json.loads(buffer)
                buffer = ""
                _push_candidate(msg)
            except json.JSONDecodeError:
                pass

            if (first_ts is not None) and (time.time() - first_ts >= CANDIDATE_WINDOW_SEC) and (not busy):
                target = pick_leftmost_and_clear()
                first_ts = None
                if target is not None:
                    p_cam = target["p_cam"]
                    uv    = target.get("uv")
                    busy = True
                    try:
                        # 현재 스캔 포즈(=초기 시퀀스 이후 포즈) 기준으로 경로 생성
                        approach_mdeg, target_mdeg, p_base = plan_from_camera_point(p_cam, curr_pose_base_m_deg=SCAN_POSE_M_DEG)
                        print(f"[SERVER] SELECT LEFTMOST → p_cam={p_cam}, uv={uv}, p_base={p_base}")
                        robot.task_move_to(mdeg_to_mmrad(approach_mdeg)); wait_idle(robot)
                        robot.task_move_to(mdeg_to_mmrad(target_mdeg));   wait_idle(robot)
                        robot.task_move_to(mdeg_to_mmrad(SCAN_POSE_M_DEG)); wait_idle(robot)
                        processed_cam.append(p_cam.copy())
                        if uv is not None:
                            processed_uv.append(uv)
                        try: conn.sendall(b"ACK_DONE\n")
                        except: pass
                        print("[SERVER] done → back to scan pose")
                    finally:
                        busy = False

if __name__ == "__main__":
    main()
