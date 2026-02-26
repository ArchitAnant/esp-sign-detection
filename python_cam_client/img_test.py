import socket
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


TARGET_ADDR: Tuple[str, int] = ("10.10.10.10", 5005) # Change according to ESP IP
FRAME_SHAPE: Tuple[int, int] = (28, 28) 

INPUT_SCALE = 0.007843137718737125
INPUT_ZERO_POINT = -1


class PerImageNormalize(object):
    """Normalize each image by its own mean/std to mimic training setup."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std if std > 0 else tensor - mean


val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(), # Scales to [0.0, 1.0]
    transforms.Normalize(mean=[0.5], std=[0.5]) # Shifts to [-1.0, 1.0]
])

def extract_bbox(hand_landmarks, frame_shape, padding: int = 40) -> Tuple[int, int, int, int]:
    height, width, _ = frame_shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(int(min(xs) * width) - padding, 0)
    x_max = min(int(max(xs) * width) + padding, width)
    y_min = max(int(min(ys) * height) - padding, 0)
    y_max = min(int(max(ys) * height) + padding, height)

    return x_min, y_min, x_max, y_max


def preprocess_roi(roi_bgr: np.ndarray) -> torch.Tensor:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    return val_transform(pil_image).unsqueeze(0)


def quantize_tensor(tensor: torch.Tensor) -> np.ndarray:
    np_input = tensor.cpu().numpy()
    np_input = np.transpose(np_input, (0, 2, 3, 1))  # NCHW -> NHWC
    np_input = np_input / INPUT_SCALE + INPUT_ZERO_POINT
    return np.clip(np_input, -128, 127).astype(np.int8)


def show_debug_view(tensor: torch.Tensor) -> None:
    debug_np = tensor.squeeze().detach().cpu().numpy()
    debug_np = debug_np - debug_np.min()
    denom = debug_np.max()
    if denom == 0:
        denom = 1.0
    debug_np = (debug_np / denom * 255).astype(np.uint8)
    cv2.imshow("What ESP32 Sees", cv2.resize(debug_np, (200, 200), interpolation=cv2.INTER_NEAREST))


def build_payload_from_roi(roi_bgr: np.ndarray) -> bytes:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    header = bytes([0xAA, 0xBB])
    return header + resized.tobytes()


def detect_hand_roi(frame: np.ndarray, hands) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    bbox = extract_bbox(hand_landmarks, frame.shape)
    x_min, y_min, x_max, y_max = bbox
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None
    return roi, bbox


def send_roi_frame(roi_bgr: np.ndarray, sock: socket.socket) -> None:
    payload = build_payload_from_roi(roi_bgr)
    sent = sock.sendto(payload, TARGET_ADDR)
    print(f"Sent ROI frame {FRAME_SHAPE[::-1]} -> {sent} bytes")


def send_white_square(sock: socket.socket) -> None:
    """Send a 28x28 all-255 frame without touching the camera."""
    # white_image = np.full((FRAME_SHAPE[1], FRAME_SHAPE[0]), 255, dtype=np.uint8)
    # header = bytes([FRAME_SHAPE[0], FRAME_SHAPE[1]])
    # payload = header + white_image.tobytes()
    # sent = sock.sendto(payload, TARGET_ADDR)
    # Create a 28x28 white image
    img = np.full((28, 28), 255, dtype=np.uint8)

    # Convert to raw bytes (Length must be exactly 784)
    pixel_bytes = img.tobytes() 

    # Add a simple 2-byte header if your ESP32 expects it (e.g., 0xAA 0xBB)
    header = bytes([0xAA, 0xBB])
    packet = header + pixel_bytes

    sent = sock.sendto(packet, (TARGET_ADDR[0], 5005))
    print(f"Sent white square -> {sent} bytes")


def run_camera_loop(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Press 'c' to capture & send, 'q' to quit.")

    mp_hands = mp.solutions.hands

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            roi_for_send: Optional[np.ndarray] = None
            bbox_for_draw: Optional[Tuple[int, int, int, int]] = None
            status_text = "Waiting for hand..."
            status_color = (0, 255, 255)

            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Unable to read frame from camera")

                frame = cv2.flip(frame, 1)
                roi, bbox = detect_hand_roi(frame, hands)

                if roi is not None and bbox is not None:
                    roi_for_send = roi
                    bbox_for_draw = bbox
                    status_text = "Hand detected"
                    status_color = (0, 255, 0)
                else:
                    roi_for_send = None
                    bbox_for_draw = None
                    status_text = "No hand detected"
                    status_color = (0, 0, 255)

                preview = frame.copy()
                if bbox_for_draw is not None:
                    x_min, y_min, x_max, y_max = bbox_for_draw
                    cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv2.putText(
                    preview,
                    "Press 'c' to send ROI",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    status_text,
                    (10, preview.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                )

                cv2.imshow("Frame Sender", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if roi_for_send is None:
                        print("No ROI available to send.")
                        continue
                    send_roi_frame(roi_for_send, sock)
        # send_white_square(sock)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    run_camera_loop()
