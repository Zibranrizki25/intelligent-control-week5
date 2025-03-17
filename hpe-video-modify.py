import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Buka video
video_path = "WIN_20250312_16_19_32_Pro.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi pose tubuh dengan YOLO
    results = model.predict(rgb_frame, save=False, show=False, conf=0.5)
    results_pose = model(frame)
    frame = results_pose[0].plot()  # Anotasi hasil pose


    # Deteksi pose tangan dengan MediaPipe Hands
    hand_results = hands.process(rgb_frame)

    # Tampilkan hasil deteksi tangan
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan hasil
    cv2.imshow("Pose & Hand Detection", frame)
    
    #
    if cv2.waitKey(5)  & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
