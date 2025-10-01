import os
import cv2
from ultralytics import YOLO

# === CONFIG ===
model_path = "runs/detect/bdd100k_yolov83/weights/best.pt"  # Path to your trained YOLOv8 model (detects 'traffic sign')
input_folder = "bdd100k/bdd100k/images/100k/test"  # Path to BDD100K test images
output_folder = "cropped_signs_for_classifier/"  # Where to save the cropped signs
target_label = "traffic sign"
confidence_threshold = 0.3  # Optional filter for low-confidence detections
padding = 5  # Pixels to pad around bounding box

# === SETUP ===
os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_path)

# === PROCESS ALL IMAGES ===
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Run YOLOv8 inference
    results = model(image)[0]

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # Skip non-traffic sign or low confidence
        if label.lower() != target_label or conf < confidence_threshold:
            continue

        # Get and clip bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Crop and save
        cropped_sign = image[y1:y2, x1:x2]
        output_name = f"{os.path.splitext(image_file)[0]}_sign_{i}.jpg"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, cropped_sign)

        print(f"[✓] Saved cropped sign: {output_name}")

print("✅ All traffic sign crops saved.")
