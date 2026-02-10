import socket
import numpy as np

ESP_IP = "192.168.4.1"
ESP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

FRAME_SIZE = 4096
CHUNK_SIZE = 1024
# divide by 8 -> 512
# divide by 4 -> 1024
NUM_CHUNKS = FRAME_SIZE // CHUNK_SIZE

frame_id = 0

def crc8(data: bytes, poly=0x07):
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1 ) & 0xFF
    return crc

def send_tensor(img_tensor: np.ndarray):
    global frame_id
    flat = img_tensor.flatten().astype(np.int8)

    for chunk_id in range(NUM_CHUNKS):
        start = chunk_id * CHUNK_SIZE
        end = start + CHUNK_SIZE
        payload = flat[start:end].tobytes()

        crc = crc8(payload)

        packet = bytes([
            frame_id,
            chunk_id,
            NUM_CHUNKS
        ]) + payload + bytes([crc])

        sock.sendto(packet, (ESP_IP, ESP_PORT))

    frame_id = (frame_id + 1) & 0xFF
    