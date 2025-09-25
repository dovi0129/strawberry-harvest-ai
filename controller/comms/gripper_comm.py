# comms/gripper_comm.py
import time
import logging
from typing import Optional
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

logger = logging.getLogger(__name__)

class GripperController:
    # Control table (XM430-W350-T, Protocol 2.0)
    ADDR_TORQUE_ENABLE    = 64
    ADDR_GOAL_POSITION    = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112

    OPEN_POSITION  = 1850  # 프로젝트 기구에 맞게 조정
    CLOSE_POSITION = 2250

    TORQUE_ENABLE  = 1
    TORQUE_DISABLE = 0

    def __init__(
        self,
        device_name: str = "/dev/ttyUSB0",
        baudrate:   int  = 1_000_000,
        protocol_ver: float = 2.0,
        motor_id:   int  = 15,
        action_delay: float = 0.2,
        profile_velocity: Optional[int] = 50,
    ):
        self.device_name  = device_name
        self.baudrate     = baudrate
        self.protocol_ver = float(protocol_ver)
        self.motor_id     = int(motor_id)
        self.delay        = float(action_delay)

        self.portHandler   = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(self.protocol_ver)
        self.connected     = False

        try:
            if not self.portHandler.openPort():
                raise RuntimeError(f"openPort failed: {self.device_name}")
            if not self.portHandler.setBaudRate(self.baudrate):
                raise RuntimeError(f"setBaudRate failed: {self.baudrate}")

            # 1) PING으로 통신 확인
            model, comm, err = self.packetHandler.ping(self.portHandler, self.motor_id)
            if comm != COMM_SUCCESS or err != 0:
                raise RuntimeError(f"Ping failed: id={self.motor_id} comm={comm} err={err}")
            logger.info(f"[DXL] ID={self.motor_id} model={model}")

            # 2) (선택) 속도 낮추기
            if profile_velocity is not None:
                comm, err = self.packetHandler.write4ByteTxRx(
                    self.portHandler, self.motor_id, self.ADDR_PROFILE_VELOCITY, int(profile_velocity)
                )
                if comm != COMM_SUCCESS or err != 0:
                    logger.warning(f"Set PROFILE_VELOCITY fail: comm={comm} err={err}")

            # 3) 토크 ON
            comm, err = self.packetHandler.write1ByteTxRx(
                self.portHandler, self.motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            if comm != COMM_SUCCESS or err != 0:
                raise RuntimeError(f"Torque enable failed: comm={comm}, err={err}")

            self.connected = True

        except Exception:
            try:
                self.portHandler.closePort()
            except Exception:
                pass
            raise

    def _write_position(self, pos: int) -> None:
        if not self.connected:
            raise RuntimeError("GripperController: not connected")
        if not isinstance(pos, int):
            raise ValueError(f"Position must be int, got {type(pos)}")
        if not (0 <= pos <= 4095):
            raise ValueError(f"Goal position out of range: {pos}")

        comm, err = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.motor_id, self.ADDR_GOAL_POSITION, pos
        )
        if comm != COMM_SUCCESS or err != 0:
            raise RuntimeError(f"Write position failed: comm={comm}, err={err}")
        time.sleep(self.delay)

    # ### 수정된 부분: open/close 메소드 추가 ###
    def open(self) -> None:
        """그리퍼를 엽니다."""
        self._write_position(self.OPEN_POSITION)

    def close(self) -> None:
        """그리퍼를 닫습니다."""
        self._write_position(self.CLOSE_POSITION)
    # ######################################

    def get_current_position(self) -> int:
        if not self.connected:
            raise RuntimeError("GripperController: not connected")
        val, comm, err = self.packetHandler.read4ByteTxRx(
            self.portHandler, self.motor_id, self.ADDR_PRESENT_POSITION
        )
        if comm != COMM_SUCCESS or err != 0:
            raise RuntimeError(f"Read position failed: comm={comm}, err={err}")
        return int(val)

    def disconnect(self) -> None:
        if self.connected:
            try:
                # 토크 OFF
                self.packetHandler.write1ByteTxRx(
                    self.portHandler, self.motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                )
            finally:
                self.portHandler.closePort()
                self.connected = False

# --- 아래 내용을 gripper_comm.py 맨 아래에 추가하세요 ---
import atexit
import os

# 환경변수로 기본값 덮어쓰기(원하면 사용)
_DXL_DEV   = os.getenv("DXL_DEVICE", "/dev/ttyUSB0")
_DXL_BAUD  = int(os.getenv("DXL_BAUD", "1000000"))
_DXL_ID    = int(os.getenv("DXL_ID", "15"))

# lazy singleton
_GRIPPER: Optional[GripperController] = None

def _get_gripper() -> GripperController:
    global _GRIPPER
    if _GRIPPER is None:
        # 필요하면 프로파일 속도나 포트 이름을 여기서 바꿔주세요
        _GRIPPER = GripperController(
            device_name=_DXL_DEV,
            baudrate=_DXL_BAUD,
            motor_id=_DXL_ID,
            profile_velocity=50,
        )
    return _GRIPPER

def gripper_open(block: bool = False, tol: int = 10, timeout: float = 1.0) -> None:
    """모듈 레벨 오픈 API (mid_to_move_jetson.py에서 바로 호출 가능)"""
    g = _get_gripper()
    g.open()
    if block:
        _wait_until_reached(g, g.OPEN_POSITION, tol=tol, timeout=timeout)

def gripper_close(block: bool = False, tol: int = 10, timeout: float = 1.0) -> None:
    """모듈 레벨 클로즈 API"""
    g = _get_gripper()
    g.close()
    if block:
        _wait_until_reached(g, g.CLOSE_POSITION, tol=tol, timeout=timeout)

def _wait_until_reached(g: GripperController, goal: int, tol: int = 10, timeout: float = 1.0) -> None:
    """원하면 블로킹 동작으로 목표 각도까지 기다림(옵션)"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            pos = g.get_current_position()
            if abs(pos - goal) <= tol:
                return
        except Exception:
            break
        time.sleep(0.02)

def _shutdown():
    global _GRIPPER
    try:
        if _GRIPPER is not None:
            _GRIPPER.disconnect()
    except Exception:
        pass
    _GRIPPER = None

atexit.register(_shutdown)
# --- 추가 끝 ---
