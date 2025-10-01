import os
import json
from PIL import Image

# === CONFIGURE THESE PATHS ===
ANNOTATION_DIR = r'mtsd_annotation\mtsd_v2_fully_annotated\annotations'  # Folder with .json files
IMAGE_ROOTS = [
    r'mtsd_train.0\images',
    r'mtsd_train.1\images',
    r'mtsd_train.2\images',
    r'mtsd_val\images',
    r'mtsd_test\images'
]
OUTPUT_DIR = r'cropped_mtsd_signs'

# === TARGET LABELS FOR YOUR RESEARCH ===
TARGET_LABELS = {
    'warning--pedestrians-crossing--g4',
    'warning--curve-left--g2',
    'warning--road-bump--g2',
    'warning--slippery-road-surface--g1',
    'warning--children--g2',
    'regulatory--stop--g1',
    'regulatory--yield--g1',
    'regulatory--priority-road--g4',
    'regulatory--no-entry--g1',
    'regulatory--no-left-turn--g1',
    'regulatory--no-u-turn--g1',
    'regulatory--no-parking--g1',
    'regulatory--no-stopping--g15',
    'information--pedestrians-crossing--g1',
    'information--parking--g1',
    'information--tram-bus-stop--g2',
    'regulatory--keep-right--g1',
    'regulatory--maximum-speed-limit-40--g1',
    'regulatory--go-straight--g1',
}

# === HELPER: Find the matching image in one of the roots ===
def find_image_path(image_key):
    for root in IMAGE_ROOTS:
        path = os.path.join(root, f'{image_key}.jpg')
        if os.path.exists(path):
            return path
    return None

# === Create folders ===
for label in TARGET_LABELS | {'other-sign'}:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# === Loop through all annotations ===
for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith('.json'):
        continue

    image_key = filename.replace('.json', '')
    annotation_path = os.path.join(ANNOTATION_DIR, filename)
    image_path = find_image_path(image_key)
    if not image_path:
        continue

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    try:
        with Image.open(image_path) as img:
            for obj in data.get("objects", []):
                label = obj["label"]
                if label not in TARGET_LABELS and label != "other-sign":
                    continue

                bbox = obj["bbox"]
                crop = img.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))

                save_path = os.path.join(OUTPUT_DIR, label, f'{image_key}_{obj["key"]}.jpg')
                crop.save(save_path)
    except Exception as e:
        print(f"Error with {image_path}: {e}")
