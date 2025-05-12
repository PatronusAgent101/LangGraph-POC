import cv2

img1_gray = cv2.imread("handwritten_crop.jpg", cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread("printed_crop.jpg", cv2.IMREAD_GRAYSCALE)

##
import numpy as np

def match_background(img_src, target_mean):
    current_mean = np.mean(img_src)
    shift = target_mean - current_mean
    img_out = np.clip(img_src + shift, 0, 255).astype(np.uint8)
    return img_out

mean1 = np.mean(img1_gray)
mean2 = np.mean(img2_gray)
target_mean = (mean1 + mean2) / 2

img1_norm = match_background(img1_gray, target_mean)
img2_norm = match_background(img2_gray, target_mean)

##

concatenated = cv2.hconcat([img1_norm, img2_norm])
cv2.imwrite("final_crop.jpg", concatenated)
