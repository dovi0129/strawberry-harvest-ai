# Strawberry Robot Project

딥러닝 기반 딸기 탐지와 로봇팔 제어를 통해 자동 수확을 수행하는 프로젝트입니다.  
This project integrates strawberry detection (YOLOv5n + UNet) with robot arm (Indy7) and gripper (OpenMANIPULATOR-X) control.

---

## Project Structure (프로젝트 구조)

```

strawberry\_robot\_project/
├── controller/
│   ├── comms/
│   │   ├── gripper\_comm.py       # Gripper control (Dynamixel SDK)
│   │   ├── mid\_to\_move\_jetson.py # Robot arm + gripper main control server
│   │   └── socket\_server.py      # Simple socket server (for debugging)
│
├── strawberry\_detection\_final/
│   ├── dl/                       # Deep learning models (YOLOv5n, UNet)
│   ├── util/                     # Utility functions
│   ├── detection\_pre.py          # Preprocessing
│   ├── detection.py              # Detection main script
│   ├── requirements.txt          # Dependencies
│   └── socket\_sender.py          # Socket client (send detection results)

````

---

## Environment (실행 환경)

- Robot Arm: **Neuromeka Indy7** (IndyDCP2 SDK)
- Gripper: **OpenMANIPULATOR-X** (Dynamixel XM430, U2D2 interface)
- Camera: **Intel RealSense D455**
- Board: **Jetson Orin Nano / Jetson Nano**
- Libraries: `pyrealsense2`, `torch`, `opencv-python`, `dynamixel-sdk`, `neuromeka`

---

## Execution (실행 방법)

1. **Run robot control server (Jetson)**  
   로봇 제어 서버 실행
   ```bash
   cd controller/comms
   python mid_to_move_jetson.py


* Socket server opens at `127.0.0.1:9999`
* Receives JSON (mid/left/right/angle) from detection
* Executes motion: move → gripper close/open → back to scan pose
* Sends `ACK_DONE` after completion

2. **Run detection module (with camera)**
   딸기 탐지 실행

   ```bash
   cd strawberry_detection_final
   python detection.py
   ```

   * Captures RGB-D frames from RealSense
   * Runs YOLOv5n + UNet for detection and segmentation
   * Classifies maturity and extracts picking points
   * Sends JSON message to control server

---

## Data Flow (데이터 흐름)

1. `detection.py` → 딸기 탐지 결과 전송 (JSON with left/right/mid/angle)
2. `mid_to_move_jetson.py` → 좌표 수신 후 로봇팔 이동 + 그리퍼 제어
3. 동작 완료 후 `ACK_DONE` 반환
4. Detection continues to next strawberry

---

## Protocol (프로토콜 정의)

* **BUSY** : Robot is working, cannot accept new command
* **SKIP\_DUP** : Duplicate or too close to previous target
* **ACCEPT** : Command accepted, executing motion
* **ACK\_DONE** : Task completed successfully

---

## Example Logs (실행 로그 예시)

**Detection side**

```
[SOCKET] Connected to 127.0.0.1:9999
[frame 42] ripe_id=3 depth=15.2cm
[SOCKET] Sent: {"left": {...}, "right": {...}, "mid": {...}, "angle": -12.3}
```

**Robot control side**

```
[SERVER] Listening on 127.0.0.1:9999 ...
[SERVER] Connected by ('127.0.0.1', 35214)
[SERVER] p_cam(m): [0.325 -0.008 0.390]  p_base(m): [...]
[GRIPPER] Closing gripper (picking)...
[GRIPPER] Opening gripper (placing)...
[SERVER] Cycle done. Back to scan pose.
```

---

## Collaboration Guide (협업 가이드)

* **Detection team**: Maintain and improve `strawberry_detection_final`
* **Control team**: Maintain `controller/comms`
* Use GitHub for version control (code and dataset managed separately)
* Communication between modules is based on **JSON over TCP with newline delimiter**

