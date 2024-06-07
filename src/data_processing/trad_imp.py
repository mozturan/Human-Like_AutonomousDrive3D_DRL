import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from process import *

data_dir = '/home/o/Documents/donkeycar_rl/data/generated_track_human'
images_org, originals = load_data(data_dir)

images = rgb_to_grayscale(images_org)
images = blur_images(images)
edges = cv2.Canny(np.uint8(images[0]*255), 50, 150)

plt.imshow(edges)
plt.show()

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),  # Bottom left
        (width , height),  # Bottom right
        (width * 0.55, height*0.1 ),  # Top right
        (width * 0.45, height*0.1 )   # Top left    
        ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


roi_edges = region_of_interest(edges)

plt.imshow(roi_edges)
plt.show()

lines = cv2.HoughLinesP(roi_edges, rho=1, 
                        theta=np.pi/180, 
                        threshold=15, 
                        minLineLength=10, 
                        maxLineGap=20)

line_features = []
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            line_features.append([x1, y1, x2, y2, angle, length])

line_features = np.array(line_features)

# Visualize the detected lane lines
line_image = np.zeros_like(images[0])
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Combine with original image
combo_image = cv2.addWeighted(images[0], 0.0, line_image, 1, 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow((rgb_to_grayscale(images_org[0])), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Detected Lane Lines")
plt.imshow((line_image), cmap='gray')

plt.show()