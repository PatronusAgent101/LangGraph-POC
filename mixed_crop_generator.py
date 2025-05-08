import os 
import random
import cv2
import numpy as np
import math
from glob import glob

# Paths
handwritten_dir = "train_data/handwritten"
printed_dir = "train_data/printed"
output_dir = "train_data/mixedcrop"
os.makedirs(output_dir, exist_ok=True)

# Params
mixed_count = 1000
imgH, imgW, imgC = 48, 192, 3  # PaddleOCR format

# Get image lists
hand_imgs = glob(os.path.join(handwritten_dir, "*.jpg"))
print_imgs = glob(os.path.join(printed_dir, "*.jpg"))

def preprocess_for_saving(img, imgH=48, imgW=192, imgC=3):
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized = cv2.resize(img, (resized_w, imgH)).astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255.0  # CHW
    resized = (resized - 0.5) / 0.5

    padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padded[:, :, :resized_w] = resized

    # Convert back to image format: [0, 255] and HWC
    restored = padded * 0.5 + 0.5  # De-normalize
    restored = (restored * 255).clip(0, 255).astype("uint8")
    restored = restored.transpose(1, 2, 0)  # HWC
    return restored

def create_mixed_crop(hand_img_path, print_img_path, mode='horizontal'):
    img1 = cv2.imread(hand_img_path)
    img2 = cv2.imread(print_img_path)

    if img1 is None or img2 is None:
        return None

    # Resize for mixing
    if mode == 'horizontal':
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (img1.shape[1], h))
        img2 = cv2.resize(img2, (img2.shape[1], h))
        mixed = np.hstack((img1[:, :img1.shape[1] // 2], img2[:, img2.shape[1] // 2:]))
    else:
        w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, img1.shape[0]))
        img2 = cv2.resize(img2, (w, img2.shape[0]))
        mixed = np.vstack((img1[:img1.shape[0] // 2, :], img2[img2.shape[0] // 2:, :]))

    return preprocess_for_saving(mixed, imgH, imgW, imgC)

# Generate and save mixed images
for i in range(mixed_count):
    hand_img = random.choice(hand_imgs)
    print_img = random.choice(print_imgs)
    mode = random.choice(['horizontal', 'vertical'])

    mixed_img = create_mixed_crop(hand_img, print_img, mode=mode)
    if mixed_img is not None:
        save_path = os.path.join(output_dir, f"mixed_{i:04d}.jpg")
        cv2.imwrite(save_path, mixed_img)

print(f"✅ Generated and saved {mixed_count} preprocessed mixed images to: {output_dir}")
