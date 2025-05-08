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
mixed_count = 1000

# Get image lists
hand_imgs = glob(os.path.join(handwritten_dir, "*.jpg"))
print_imgs = glob(os.path.join(printed_dir, "*.jpg"))

def get_dominant_border_color(img, side='left', sample_rows=5):
    """Estimate the background color from the left or right edge of the image."""
    h, w, _ = img.shape
    if side == 'left':
        border = img[:sample_rows, :2, :]
    else:
        border = img[:sample_rows, -2:, :]
    avg_color = np.mean(border.reshape(-1, 3), axis=0)
    return tuple(int(c) for c in avg_color)

def pad_to_height(img, target_height, bg_color):
    h, w, c = img.shape
    if h == target_height:
        return img
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0,
                               borderType=cv2.BORDER_CONSTANT, value=bg_color)

def create_mixed_crop(left_img_path, right_img_path):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    if left_img is None or right_img is None:
        return None

    target_height = max(left_img.shape[0], right_img.shape[0])

    bg_color = get_dominant_border_color(left_img, side='left')

    left_padded = pad_to_height(left_img, target_height, bg_color)
    right_padded = pad_to_height(right_img, target_height, bg_color)

    mixed = np.hstack((left_padded, right_padded))
    return mixed

# Create mixed samples
for i in range(mixed_count):
    hand_img = random.choice(hand_imgs)
    print_img = random.choice(print_imgs)

    if i % 2 == 0:
        # HANDWRITTEN left, PRINTED right
        mixed_img = create_mixed_crop(hand_img, print_img)
    else:
        # PRINTED left, HANDWRITTEN right
        mixed_img = create_mixed_crop(print_img, hand_img)

    if mixed_img is not None:
        save_path = os.path.join(output_dir, f"mixed_{i:04d}.jpg")
        cv2.imwrite(save_path, mixed_img)

print(f"✅ Generated {mixed_count} visually consistent mixed crops at: {output_dir}")
