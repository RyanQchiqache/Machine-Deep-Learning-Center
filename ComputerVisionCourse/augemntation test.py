
import cv2
import numpy as np
import random

# Parameters
image_path = '/ComputerVisionCourse/Exercise04/2012-04-26-Muenchen-Tunnel_4K0G0010.jpg'
crop_size = 200  # square crop
angle = random.uniform(0, 360)  # random rotation angle in degrees

# Load image
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Choose a random center (ensure crop fits)
margin = crop_size // 2 + 1

for i in range(10):

    cx = random.randint(margin, w - margin)
    cy = random.randint(margin, h - margin)

    # Define 4 corners in local frame
    half = crop_size / 2
    pts = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ])

    # Rotation
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated_pts = np.dot(pts, R.T) + [cx, cy]

    # Get perspective transform and crop
    src_pts = rotated_pts.astype(np.float32)
    dst_pts = np.array([
        [0, 0],
        [crop_size-1, 0],
        [crop_size-1, crop_size-1],
        [0, crop_size-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    crop = cv2.warpPerspective(img, M, (crop_size, crop_size))

    # Draw the crop region on the original image (make a copy for display)
    img_vis = img.copy()
    poly_pts = rotated_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_vis, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow("Original Image with Crop Region", img_vis)
    cv2.imshow("Cropped Image", crop)
    cv2.waitKey(0)