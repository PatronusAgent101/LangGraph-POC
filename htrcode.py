import os
import random
import cv2
import numpy as np
from glob import glob

# Paths
handwritten_dir = "train_data/handwritten"
printed_dir = "train_data/printed"
output_dir = "train_data/mixedcrop"
os.makedirs(output_dir, exist_ok=True)

# Params
mixed_count = 1000  # how many mixed samples to generate
resize_dim = (224, 224)  # resizing each crop to this size

# Get image lists
hand_imgs = glob(os.path.join(handwritten_dir, "*.jpg"))
print_imgs = glob(os.path.join(printed_dir, "*.jpg"))

def create_mixed_crop(hand_img_path, print_img_path, mode='horizontal'):
    img1 = cv2.imread(hand_img_path)
    img2 = cv2.imread(print_img_path)

    if img1 is None or img2 is None:
        return None

    img1 = cv2.resize(img1, resize_dim)
    img2 = cv2.resize(img2, resize_dim)

    if mode == 'horizontal':
        mixed = np.hstack((img1[:, :resize_dim[0]//2], img2[:, resize_dim[0]//2:]))
    else:
        mixed = np.vstack((img1[:resize_dim[1]//2, :], img2[resize_dim[1]//2:, :]))

    return mixed

# Create mixed samples
for i in range(mixed_count):
    hand_img = random.choice(hand_imgs)
    print_img = random.choice(print_imgs)
    mode = random.choice(['horizontal', 'vertical'])

    mixed_img = create_mixed_crop(hand_img, print_img, mode=mode)
    if mixed_img is not None:
        save_path = os.path.join(output_dir, f"mixed_{i:04d}.jpg")
        cv2.imwrite(save_path, mixed_img)

print(f"Generated {mixed_count} synthetic mixed crops at: {output_dir}")


################## METHOD 2

import os
import random
import cv2
import numpy as np
from glob import glob

# Config
handwritten_dir = "train_data/handwritten"
printed_dir = "train_data/printed"
output_dir = "train_data/mixedcrop"
label_file = "train_data/mixed_list.txt"
os.makedirs(output_dir, exist_ok=True)

resize_dim = (224, 224)
mixed_count = 1000
label_id = 2  # Class label for MIXEDCROP

# Augmentation functions
def add_gaussian_noise(img, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

def apply_blur(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)

def adjust_brightness(img, factor=0.8 + 0.4 * random.random()):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

# Mixed crop creation
def create_mixed_crop(hand_img_path, print_img_path, mode='horizontal'):
    img1 = cv2.imread(hand_img_path)
    img2 = cv2.imread(print_img_path)

    if img1 is None or img2 is None:
        return None

    img1 = cv2.resize(img1, resize_dim)
    img2 = cv2.resize(img2, resize_dim)

    # Random augment one of them
    if random.random() < 0.5:
        img1 = add_gaussian_noise(img1)
    if random.random() < 0.5:
        img2 = apply_blur(img2)
    if random.random() < 0.5:
        img1 = adjust_brightness(img1)
    if random.random() < 0.5:
        img2 = adjust_brightness(img2)

    # Stitch
    if mode == 'horizontal':
        mixed = np.hstack((img1[:, :resize_dim[0] // 2], img2[:, resize_dim[0] // 2:]))
    else:
        mixed = np.vstack((img1[:resize_dim[1] // 2, :], img2[resize_dim[1] // 2:, :]))

    return mixed

# Get source images
hand_imgs = glob(os.path.join(handwritten_dir, "*.jpg"))
print_imgs = glob(os.path.join(printed_dir, "*.jpg"))

# Create mixed images and label file
with open(label_file, "w") as f:
    for i in range(mixed_count):
        hand_img = random.choice(hand_imgs)
        print_img = random.choice(print_imgs)
        mode = random.choice(['horizontal', 'vertical'])

        mixed_img = create_mixed_crop(hand_img, print_img, mode=mode)
        if mixed_img is not None:
            save_path = os.path.join(output_dir, f"mixed_{i:04d}.jpg")
            cv2.imwrite(save_path, mixed_img)
            f.write(f"{save_path}\t{label_id}\n")

print(f"Done! {mixed_count} mixed crops saved to '{output_dir}' and labels in '{label_file}'")


###### COMBINE

import random
from pathlib import Path

# Config: update paths if needed
handwritten_file = "train_data/handwritten_list.txt"
printed_file     = "train_data/printed_list.txt"
mixed_file       = "train_data/mixed_list.txt"

train_output = "train_data/train_list.txt"
val_output   = "train_data/val_list.txt"

val_split_ratio = 0.2  # 20% for validation

# Load data
def load_labels(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

handwritten = load_labels(handwritten_file)
printed = load_labels(printed_file)
mixed = load_labels(mixed_file)

# Balance data: use the minimum count from the 3 classes
min_count = min(len(handwritten), len(printed), len(mixed))
print(f"Balancing all classes to {min_count} samples.")

handwritten = random.sample(handwritten, min_count)
printed = random.sample(printed, min_count)
mixed = random.sample(mixed, min_count)

# Combine and shuffle
combined = handwritten + printed + mixed
random.shuffle(combined)

# Split
val_size = int(len(combined) * val_split_ratio)
val_set = combined[:val_size]
train_set = combined[val_size:]

# Save
Path(train_output).write_text("\n".join(train_set) + "\n")
Path(val_output).write_text("\n".join(val_set) + "\n")

print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
print(f"Saved to:\n  {train_output}\n  {val_output}")

### VISUALIZAION

import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Config
label_file = "train_data/train_list.txt"
class_names = {0: "HANDWRITTEN", 1: "PRINTED", 2: "MIXED"}
samples_per_class = 5

# Load all labeled samples
with open(label_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Group by class
class_dict = {0: [], 1: [], 2: []}
for line in lines:
    path, label = line.split('\t')
    label = int(label)
    if label in class_dict:
        class_dict[label].append(path)

# Randomly pick samples
picked = {
    cls: random.sample(paths, min(samples_per_class, len(paths)))
    for cls, paths in class_dict.items()
}

# Plot
fig, axs = plt.subplots(len(class_names), samples_per_class, figsize=(15, 6))
for row, (cls, img_paths) in enumerate(picked.items()):
    for col, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[row, col].imshow(img)
        axs[row, col].axis('off')
        if col == 0:
            axs[row, col].set_title(class_names[cls], fontsize=12)
        else:
            axs[row, col].set_title("")

plt.tight_layout()
plt.show()
