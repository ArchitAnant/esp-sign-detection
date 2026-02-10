import socket
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

TARGET_ADDR: Tuple[str, int] = ("10.90.115.71", 5005)
FRAME_SHAPE: Tuple[int, int] = (28, 28)  # (width, height)
RECV_TIMEOUT = 5.0
RECV_BUFFER = 2048
OUTPUT_IMAGE = Path("received_frame.png")


def capture_frame(camera_index: int = 0) -> np.ndarray:
    """Grab a single frame from the given camera index."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to capture frame from camera")
        return frame
    finally:
        cap.release()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert to grayscale and resize to the expected input shape."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FRAME_SHAPE, interpolation=cv2.INTER_AREA)
    return resized


def build_payload(image_28: np.ndarray) -> bytes:
    """Prefix width/height and append flattened pixel data as bytes."""
    if image_28.shape != FRAME_SHAPE[::-1]:  # cv2 returns (h, w)
        raise ValueError("Input image must be 28x28 pixels")

    flattened = image_28.astype(np.uint8).flatten()
    header = bytes([FRAME_SHAPE[0], FRAME_SHAPE[1]])
    return header + flattened.tobytes()


def receive_payload(sock: socket.socket) -> bytes:
    """Wait for a UDP response containing an encoded grayscale frame."""
    sock.settimeout(RECV_TIMEOUT)
    try:
        data, _ = sock.recvfrom(RECV_BUFFER)
        if not data:
            raise RuntimeError("Received empty payload")
        return data
    finally:
        sock.settimeout(None)


def decode_payload(payload: bytes) -> np.ndarray:
    """Decode a payload that stores width/height followed by pixel bytes."""
    if len(payload) < 2:
        raise ValueError("Payload too small to contain header")

    width, height = payload[0], payload[1]
    expected_pixels = width * height
    pixel_bytes = payload[2: 2 + expected_pixels]

    if len(pixel_bytes) != expected_pixels:
        raise ValueError("Payload does not match declared dimensions")

    return np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((height, width))


def save_image(image: np.ndarray, path: Path) -> Path:
    """Persist the received image to disk as a grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def main() -> None:
    frame = capture_frame()
    processed = preprocess_frame(frame)
    payload = build_payload(processed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, TARGET_ADDR)
        print(f"Sent frame {FRAME_SHAPE[0]}x{FRAME_SHAPE[1]} with {len(payload)} bytes")

        try:
            response = receive_payload(sock)
        except socket.timeout:
            print("Timed out waiting for response payload")
            return

        decoded = decode_payload(response)
        saved_path = save_image(decoded, OUTPUT_IMAGE)
        print(f"Saved response image to {saved_path}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
