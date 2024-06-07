import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Example image (replace with your actual image)
image = cv2.imread('road_image.jpg')  # Load an image of shape (80, 160, 3)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Grayscale Conversion
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 2. HOG Descriptor
fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

# Rescale histogram for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Visualize the HOG features
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("HOG Features")
plt.imshow(hog_image_rescaled, cmap='gray')

plt.show()