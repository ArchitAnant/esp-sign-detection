import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MAX_HAND_TENSORS = 15
FRAME_INTERVAL = 1.0 / 15.0

hand_tensors = deque(maxlen=MAX_HAND_TENSORS)
hand_tensor_array = np.empty((0, 64, 64), dtype=np.float32)

# using the default camera
cap = cv2.VideoCapture(0)

while True:
    frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Check if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get bounding box coordinates
            h, w, c = frame.shape
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            # Add padding
            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, w)
            y_max = min(y_max + pad, h)
            
            # Crop hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            
            if hand_crop.size > 0:
                # Convert to grayscale and resize
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                hand_64 = cv2.resize(gray, (64, 64))
                tensor = hand_64.astype(np.float32) / 255.0
                hand_tensors.append(tensor)
                hand_tensor_array = np.stack(hand_tensors, axis=0)
                
                cv2.imshow("Hand Crop", hand_64)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display "Hand Detected"
        cv2.putText(frame, "Hand Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No hand detected
        cv2.putText(frame, "No Hand Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(
        frame,
        f"Tensor Buffer: {len(hand_tensors)}/{MAX_HAND_TENSORS}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_duration = time.perf_counter() - frame_start
    if frame_duration < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - frame_duration)
    
cap.release()
hands.close()
cv2.destroyAllWindows()
