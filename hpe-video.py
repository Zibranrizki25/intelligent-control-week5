from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Deteksi pose pada video
cap = cv2.VideoCapture("WIN_20250312_16_19_32_Pro.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame,show=True)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()