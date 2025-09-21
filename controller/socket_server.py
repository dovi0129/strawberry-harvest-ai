import socket
import json

HOST = '127.0.0.1'
PORT = 9999 

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT}...")

        conn, addr = s.accept()
        print(f"[SERVER] Connected by {addr}")  # <-- 연결 성공 메시지
        conn.sendall(b"[SERVER] Connection established\n")  # <-- 클라이언트로도 보내줌

        with conn:
            buffer = ""
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode('utf-8')
                try:
                    message = json.loads(buffer)
                    print("[SERVER] Received JSON:")
                    print(json.dumps(message, indent=2))
                    buffer = ""
                except json.JSONDecodeError:
                    continue

if __name__ == '__main__':
    start_server()
