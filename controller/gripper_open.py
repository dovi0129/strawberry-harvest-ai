# examples/gripper_open.py
# 사용: PYTHONPATH=. python examples/gripper_open.py

import time
from comms.gripper_comm import GripperController

DEVICE   = "/dev/ttyUSB0"
MOTOR_ID = 15

if __name__ == "__main__":
    g = GripperController(device_name=DEVICE, motor_id=MOTOR_ID, profile_velocity=50)
    try:
        target = int(g.OPEN_POSITION)
        print(f"[OPEN] -> {target}")
        g._write_position(target)
        time.sleep(0.5)
        now = g.get_current_position()
        print(f"[OPEN] now={now}")
    finally:
        g.disconnect()
        print("disconnected.")
