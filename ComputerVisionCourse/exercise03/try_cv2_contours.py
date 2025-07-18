import cv2
import numpy as np
from PIL import Image

# Load and resize an image
image_path = "/Users/ryanqchiqache/PycharmProjects/Machine-Learning-Learning-Center/ComputerVisionCourse/exercise03/input/Bulldog.png"
img = Image.open(image_path).convert("RGB").resize((224, 224))
img_np = np.array(img)

# Convert to grayscale
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

# Apply thresholding
threshold_value = 100
_, binary_map = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and bounding boxes
contour_img = img_np.copy()
for i, cnt in enumerate(contours):
    if len(cnt) < 3:
        continue  # skip small/noisy contours

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(contour_img, [cnt], -1, (255, 0, 0), 1)  # draw contour in blue

    print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, points={len(cnt)}")

# Show results using OpenCV window (or save if running in notebook)
cv2.imshow("Original", img_np)
cv2.imshow("Binary Map", binary_map)
cv2.imshow("Contours + Bounding Boxes", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
