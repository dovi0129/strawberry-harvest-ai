# strawberry_detection_final/socket_sender.py

import socket
import json

class SocketSender:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.sock = None
        self.is_connected = False

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"[SOCKET] Connected to {self.host}:{self.port}")
            self.is_connected = True
        except Exception as e:
            print(f"[SOCKET] Connection failed: {e}")
            self.is_connected = False

    def send_data(self, data: dict):
        if not self.is_connected:
            # print("[SOCKET] Not connected. Attempting to reconnect...")
            # self.connect() # 주석 처리: 연결이 끊기면 계속 재시도하지 않도록 함
            if not self.is_connected:
                print("[SOCKET] Cannot send data, connection failed.")
                return

        try:
            # ### 수정된 부분: 메시지 끝에 줄바꿈(\n) 추가 ###
            message = json.dumps(data) + "\n"
            self.sock.sendall(message.encode('utf-8'))
            print(f"[SOCKET] Sent: {json.dumps(data)}")

        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"[SOCKET] Connection lost: {e}. Please restart the client.")
            self.is_connected = False
        except Exception as e:
            print(f"[SOCKET] Error sending data: {e}")
            self.is_connected = False

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            self.is_connected = False
            print("[SOCKET] Connection closed.")