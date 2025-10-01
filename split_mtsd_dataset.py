import os
import shutil
import random

SOURCE_DIR = r'cropped_mtsd_signs'
DEST_DIR = r'cnn_sign_dataset'
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test

# Create split folders
for split in ['train', 'val', 'test']:
    for label in os.listdir(SOURCE_DIR):
        os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)

# Split and copy images
for label in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label)
    images = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    for i, img in enumerate(images):
        src = os.path.join(label_path, img)
        if i < n_train:
            dst = os.path.join(DEST_DIR, 'train', label, img)
        elif i < n_train + n_val:
            dst = os.path.join(DEST_DIR, 'val', label, img)
        else:
            dst = os.path.join(DEST_DIR, 'test', label, img)
        shutil.copy(src, dst)

print("Dataset split complete.")
