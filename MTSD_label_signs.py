import os
import json
from collections import Counter

# === SET THIS PATH TO YOUR ANNOTATION DIRECTORY ===
annotation_dir = "mtsd_annotation/mtsd_v2_fully_annotated/annotations"

# Collect all label names
all_labels = []

# Traverse all JSON files in the directory
for file in os.listdir(annotation_dir):
    if file.endswith(".json"):
        with open(os.path.join(annotation_dir, file), 'r') as f:
            data = json.load(f)
            for obj in data.get("objects", []):
                label = obj.get("label")
                if label:
                    all_labels.append(label)

# Count label frequencies
label_counts = Counter(all_labels)

# Sort by most common
sorted_counts = label_counts.most_common()

# Print result
print(f"Total unique labels: {len(sorted_counts)}\n")
print("Label Frequencies:")
for label, count in sorted_counts:
    print(f"{label}: {count}")
