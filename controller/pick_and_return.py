import math, numpy as np
import time
from neuromeka import IndyDCP2

# ---------- 유틸 ----------
def rpy_deg_to_R(rpy):
    rx, ry, rz = [math.radians(a) for a in rpy]
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[ math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx  # ZYX

def R_to_rpy_deg(R):
    sy = -R[2,0]; cy = math.sqrt(max(0.0, 1 - sy*sy))
    if cy > 1e-6:
        rx = math.atan2(R[2,1], R[2,2]); ry = math.asin(sy); rz = math.atan2(R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1]); ry = math.asin(sy); rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]

def pose_to_T(pose_m_deg):  # [x,y,z, rx,ry,rz] (m,deg)
    x,y,z, rx,ry,rz = pose_m_deg
    T = np.eye(4); T[:3,:3] = rpy_deg_to_R([rx,ry,rz]); T[:3,3] = [x,y,z]
    return T

def T_to_pose(T):           # -> [x,y,z, rx,ry,rz] (m,deg)
    x,y,z = T[:3,3].tolist()
    rx,ry,rz = R_to_rpy_deg(T[:3,:3])
    return [x,y,z, rx,ry,rz]

def h(p3):                  # [x,y,z] -> [x,y,z,1]
    return np.array([p3[0], p3[1], p3[2], 1.0], dtype=float)

def deproject(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=float)

# ---------- 카메라/딸기 입력 (예: frame 9621) ----------
depth_cm = 32.0
mid_px   = (367, 275)       # midpoint 사용
Z = depth_cm / 100.0        # -> 0.32 m

# RealSense intrinsics (네 실제 값으로 세팅!)
fx, fy = 616.0, 616.0
cx, cy = 320.0, 240.0

p_cam = deproject(mid_px[0], mid_px[1], Z, fx, fy, cx, cy)

# ---------- 현재 EE 포즈 (indy.get_task_pos() 방금 값) ----------
curr_pose_base = [0.48056710764986715, -0.10483965980679384, 0.6145961463793236,
                  89.92185168968957,    112.8833918977004,   99.8857288158322]

# ---------- 핸드아이: 카메라 -> EE ----------
T_ee_cam = np.eye(4)
T_ee_cam[:3, 3] = np.array([0.10, 0.00, 0.127])  # m
# 회전 보정이 있다면: T_ee_cam[:3,:3] = R

# ---------- p_cam -> p_base ----------
T_base_ee  = pose_to_T(curr_pose_base)
T_base_cam = T_base_ee @ T_ee_cam
p_base     = (T_base_cam @ h(p_cam))[:3]

# ---------- 목표 포즈(현재 자세 유지 + 백오프/리프트) ----------
KEEP_ORI     = True
LIFT_Z_BASE  = 0.200
BACKOFF_EE_Z = 0.200

R_target = T_base_ee[:3,:3] if KEEP_ORI else np.eye(3)

ee_plus_z_in_base = R_target[:,2]
p_base_backed     = p_base - ee_plus_z_in_base * BACKOFF_EE_Z
p_base_final      = p_base_backed.copy()
p_base_final[2]  += LIFT_Z_BASE

T_target   = np.eye(4); T_target[:3,:3] = R_target; T_target[:3,3] = p_base_final
T_approach = T_target.copy(); T_approach[2,3] += 0.05

approach_pose = T_to_pose(T_approach)  # [m,deg]
target_pose   = T_to_pose(T_target)    # [m,deg]

print("p_cam       :", p_cam)
print("p_base      :", p_base)
print("approach_pose:", approach_pose)
print("target_pose  :", target_pose)

# ---------- 로봇 연결/이동 ----------
IP, NAME = "192.168.0.7", "NRMK-Indy7"
indy = IndyDCP2(IP, NAME)
indy.connect()
indy.set_servo_on(True)

# 작업공간 이동 속도/가속도
indy.set_task_vel_level(2)
indy.set_task_acc_level(2)
try: indy.set_blend_radius(0.0)
except: pass

# 1) 접근
indy.task_move_to(approach_pose); indy.wait_for_move_finish()

# 2) 타겟
indy.task_move_to(target_pose);   indy.wait_for_move_finish()
time.sleep(0.2)

# 3) 접근 높이로 복귀
indy.task_move_to(approach_pose); indy.wait_for_move_finish()

# 4) 스캔 복귀 시퀀스: go_home -> joint_move_by(정면 촬영자세)
indy.go_home()
indy.wait_for_move_finish()

# joint_move_by는 "상대" 조인트 이동(단위: deg)
indy.set_joint_vel_level(2)
indy.set_joint_acc_level(2)
indy.joint_move_by([10, 15, -30, 0, 108, -35])
indy.wait_for_move_finish()

