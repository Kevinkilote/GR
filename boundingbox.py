import cv2
import os

# Configuration
image_dir = "YOLO/images/train"
label_dir = "YOLO/labels/train"
output_dir = "YOLO/preview"
class_names = ["traffic light", "traffic sign", "car"]  # Update with your class names
os.makedirs(output_dir, exist_ok=True)

# Loop through all txt files
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    # Corresponding image file
    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read label file
    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            # Convert from relative to absolute coordinates
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)

            # Draw rectangle and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[int(class_id)]
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Save visualized image
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)
    print(f"Saved preview: {output_path}")
