from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
 ret, frame = cap.read()
 frame = cv2.resize(frame, (640, 480))

 if not ret:
   break
 # Deteksi pose
 results = model(frame)
 # Tampilkan hasil
 for result in results:
    annotated_frame = result.plot() # Tambahkan anotasi pada frame
    keypoints = result.keypoints.xy.cpu().numpy()  # Ambil keypoints dari YOLOv8
    for person in keypoints:
        valid_points = {}  # Simpan titik yang valid
        for i, (x, y) in enumerate(person):
            if not np.isnan(x) and not np.isnan(y):  # Pastikan titik valid
                valid_points[i] = (int(x), int(y))
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Gambar titik
    
    # Define hand_results before using it
    hand_results = None  # Replace this with actual hand detection logic if available
    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_points = {}  # Simpan titik-titik tangan

         # Gambar titik sendi tangan
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                hand_points[i] = (x, y)  # Simpan titik
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Titik merah untuk sendi tangan

    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame),("YOLOv8 Full-Body Pose + Finger Tracking", frame) #
 if cv2.waitKey(10) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()