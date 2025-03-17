import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Konversi warna ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi pose dengan YOLOv8
    results_pose = model(frame)
    annotated_frame = results_pose[0].plot()  # Anotasi hasil pose
    
    # Deteksi tangan dengan MediaPipe
    results_hands = hands.process(frame_rgb)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose + MediaPipe Hands", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
