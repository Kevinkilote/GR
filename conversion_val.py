import json
import os
import shutil
from tqdm import tqdm

# === Paths ===
json_path = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
base_image_dir = "bdd100k/bdd100k/images/100k/val"  # Directory containing val images
output_image_dir = "YOLO/images/val"
output_label_dir = "YOLO/labels/val"

# === Create output folders ===
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# === Categories of interest ===
categories = ["traffic light", "traffic sign", "car"]
category_id_map = {name: i for i, name in enumerate(categories)}

# === Load annotations ===
with open(json_path, "r") as f:
    data = json.load(f)

# === Function to search image ===
def find_image(image_name, root_dir):
    for root, dirs, files in os.walk(root_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

# === Convert annotations ===
for item in tqdm(data, desc="Converting validation set", unit="image"):
    image_name = item["name"]

    image_input_path = find_image(image_name, base_image_dir)
    if image_input_path is None:
        print(f"⚠️ Image not found: {image_name}")
        continue

    # Copy image to YOLO/val/images/
    image_output_path = os.path.join(output_image_dir, image_name)
    if not os.path.exists(image_output_path):
        shutil.copy(image_input_path, image_output_path)

    # Create YOLO label
    label_output_path = os.path.join(output_label_dir, image_name.replace(".jpg", ".txt"))
    with open(label_output_path, "w") as label_file:
        for label in item.get("labels", []):
            category = label.get("category")
            if category not in categories:
                continue

            box = label.get("box2d", None)
            if not box:
                continue

            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            img_width, img_height = 1280, 720

            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = category_id_map[category]
            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Validation set conversion complete.")
