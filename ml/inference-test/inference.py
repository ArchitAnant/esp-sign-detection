import cv2
import torch
from PIL import Image
from torchvision import transforms
import time
import mediapipe as mp
import numpy as np
from model import TinyRes28

MODEL_PATH = "models/res_28-128_si-12.pt"
IDX_TO_CLASS = {
    0: "A",
    1: "Q",
    2: "T" 
}
TEMPERATE_SCALE = 0.7364

class PerImageNormalize(object):
    """Normalize image by its own mean and std."""
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        # Avoid division by zero
        if std > 0:
            return (tensor - mean) / std
        return tensor - mean

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    PerImageNormalize() # Replace standard Normalize with this
])

print("[+] Loading Model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyRes28(num_classes=len(IDX_TO_CLASS))
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
model.eval()
model = model.to(device)

TARGET_FPS = 15
CONFIDENCE_THRESHOLD = 0.85

def preprocess_roi(roi_bgr):
    """Crop -> PIL -> tensor pipeline for model input."""
    # Convert BGR to RGB for PIL
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    return val_transform(pil_image)


def isolate_hand_region(frame, hand_landmarks):
    """Generate a convex-hull mask around the hand and return masked frame."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = []
    for lm in hand_landmarks.landmark:
        points.append([int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])])

    if len(points) < 3:
        return frame

    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def extract_bbox(hand_landmarks, frame_shape, padding=40):
    height, width, _ = frame_shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(int(min(xs) * width) - padding, 0)
    x_max = min(int(max(xs) * width) + padding, width)
    y_min = max(int(min(ys) * height) - padding, 0)
    y_max = min(int(max(ys) * height) + padding, height)

    return x_min, y_min, x_max, y_max


def classify_gesture(roi_bgr):
    """Run the model on the provided ROI and return label + confidence."""
    processed = preprocess_roi(roi_bgr).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(processed)
        probs = torch.softmax(logits/TEMPERATE_SCALE, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())
    print(logits)
    label = IDX_TO_CLASS.get(pred_idx, "Unknown")
    return label, confidence


def run_camera_inference():
    cap = cv2.VideoCapture(0)
    
    # Set camera to 15 or 30 fps if supported to reduce lag
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
                if not ret: break

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    # masked_frame = isolate_hand_region(frame, hand_landmarks)
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

                # UI Overlay at bottom of the frame
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
                cv2.imshow("Gesture Inference", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # FPS Control
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