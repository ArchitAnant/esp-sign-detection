import time
from typing import Dict, Tuple
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms

TFLITE_MODEL_PATH = "models/gesture_model_esp32-16.tflite"

IDX_TO_CLASS: Dict[int, str] = {
    0: "A",
    1: "Q",
    2: "T"
}

TARGET_FPS = 15
CONFIDENCE_THRESHOLD = 0.05
TEMPERATE_SCALE = 0.7364


class PerImageNormalize(object):
    """Normalize each image by its own mean/std to mimic training setup."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std if std > 0 else tensor - mean


val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    PerImageNormalize(),
])

print("[+] Loading TFLite Model")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()
INPUT_SHAPE = tuple(INPUT_DETAILS[0]["shape"])
INPUT_DTYPE = INPUT_DETAILS[0]["dtype"]
input_scale, input_zero_point = INPUT_DETAILS[0]['quantization']
output_scale, output_zero_point = INPUT_DETAILS[0]['quantization']


def preprocess_roi(roi_bgr: np.ndarray) -> torch.Tensor:
    """Crop -> PIL -> tensor pipeline for model input."""
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    return val_transform(pil_image)


def prepare_tflite_input(tensor: torch.Tensor) -> np.ndarray:
    np_input = tensor.cpu().numpy()

    # NCHW -> NHWC
    np_input = np.transpose(np_input, (0, 2, 3, 1))

    # Quantize
    np_input = np_input / input_scale + input_zero_point
    np_input = np.clip(np_input, -128, 127).astype(np.int8)

    return np_input



def run_tflite_inference(model_input: np.ndarray) -> torch.Tensor:
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], model_input)
    interpreter.invoke()

    output_int8 = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])
    output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale

    return torch.from_numpy(output_float)


def classify_gesture(roi_bgr: np.ndarray):
    processed = preprocess_roi(roi_bgr).unsqueeze(0)
    model_input = prepare_tflite_input(processed)

    logits = run_tflite_inference(model_input)
    probs = torch.softmax(logits/TEMPERATE_SCALE, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1))
    confidence = float(probs[0, pred_idx])

    return IDX_TO_CLASS[pred_idx], confidence


def extract_bbox(hand_landmarks, frame_shape, padding=40):
    height, width, _ = frame_shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(int(min(xs) * width) - padding, 0)
    x_max = min(int(max(xs) * width) + padding, width)
    y_min = max(int(min(ys) * height) - padding, 0)
    y_max = min(int(max(ys) * height) + padding, height)

    return x_min, y_min, x_max, y_max


def run_camera_inference():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    target_interval = 1.0 / TARGET_FPS
    last_frame_time = time.time()
    display_text = "Waiting for hand..."
    display_color = (0, 255, 255)

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    x_min, y_min, x_max, y_max = extract_bbox(hand_landmarks, frame.shape)
                    roi = frame[y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        label, confidence = classify_gesture(roi)
                        if confidence >= CONFIDENCE_THRESHOLD:
                            display_text = f"{label}: {confidence * 100:.1f}%"
                            display_color = (0, 255, 0)
                        else:
                            display_text = "Unclassified"
                            display_color = (0, 255, 255)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    else:
                        display_text = "Hand ROI empty"
                        display_color = (0, 165, 255)
                else:
                    display_text = "No hand detected"
                    display_color = (0, 0, 255)

                baseline_y = frame.shape[0] - 20
                cv2.putText(
                    frame,
                    display_text,
                    (20, baseline_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    display_color,
                    2,
                )
                cv2.imshow("Gesture Inference (TFLite)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                last_frame_time = time.time()
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_inference()
