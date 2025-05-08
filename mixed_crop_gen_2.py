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
background_color = (255, 255, 255)  # white background

# Get image lists
hand_imgs = glob(os.path.join(handwritten_dir, "*.jpg"))
print_imgs = glob(os.path.join(printed_dir, "*.jpg"))

def pad_to_height(img, target_height, bg_color=(255, 255, 255)):
    h, w, c = img.shape
    if h == target_height:
        return img
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=bg_color)

def create_mixed_crop(hand_img_path, print_img_path):
    img1 = cv2.imread(hand_img_path)
    img2 = cv2.imread(print_img_path)

    if img1 is None or img2 is None:
        return None

    # Match taller height
    target_height = max(img1.shape[0], img2.shape[0])
    img1_padded = pad_to_height(img1, target_height, background_color)
    img2_padded = pad_to_height(img2, target_height, background_color)

    # Horizontally stack both full images
    mixed = np.hstack((img1_padded, img2_padded))
    return mixed

# Generate mixed images
for i in range(mixed_count):
    hand_img = random.choice(hand_imgs)
    print_img = random.choice(print_imgs)
    
    mixed_img = create_mixed_crop(hand_img, print_img)
    if mixed_img is not None:
        save_path = os.path.join(output_dir, f"mixed_{i:04d}.jpg")
        cv2.imwrite(save_path, mixed_img)

print(f"✅ Generated {mixed_count} full mixed crops at: {output_dir}")
