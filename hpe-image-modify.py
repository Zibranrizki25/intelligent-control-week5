import cv2
import mediapipe as mp
import os
from ultralytics import YOLO

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=50, min_detection_confidence=0.5)

# Baca gambar
image_path = "Screenshot (773).png"  # Ganti dengan path gambar
image = cv2.imread(image_path)

if image is None:
    print("Gagal membaca gambar.")
else:
    # Simpan salinan gambar asli untuk ditampilkan nanti
    original_image = image.copy()

    # Deteksi pose tubuh dengan YOLO pada gambar asli
    results_pose = model(original_image)
    image_with_pose = results_pose[0].plot()  # Gambar hasil pose

    # Konversi ke RGB hanya untuk MediaPipe
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Deteksi pose tangan dengan MediaPipe Hands
    hand_results = hands.process(rgb_image)

    # Gambar hasil deteksi tangan pada gambar asli (BGR)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_with_pose, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tentukan folder penyimpanan otomatis (sesuai YOLO)
    save_dir = "runs/pose/predict"  # Folder default YOLO
    
    # Simpan gambar hasil deteksi
    output_path = os.path.join(save_dir, "result.jpg")
    cv2.imwrite(output_path, image_with_pose)
    print(f"Hasil deteksi disimpan di: {output_path}")

    # Tampilkan hasil
    cv2.imshow("Pose & Hand Detection", image_with_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
