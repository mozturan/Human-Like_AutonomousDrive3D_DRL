import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import glob

# Load the saved LiDAR data
lidar_data = np.load('/home/o/Documents/donkeycar_rl/data/pack/generated_track_human/lidar_data.npy')  # Ensure this file exists and is correctly formatted

for i in range(len(lidar_data)):
    for k in range(len(lidar_data[i])):
        if lidar_data[i][k] < 0:
            lidar_data[i][k] = 20.0
        #normalize
        lidar_data[i][k] = lidar_data[i][k] / 20.0

    
# Load the camera images
image_files = sorted(glob.glob('/home/o/Documents/donkeycar_rl/data/pack/generated_track_human/*.jpg'))  # Adjust the pattern to match your image files
images = [Image.open(img) for img in image_files]

assert len(lidar_data) == len(images), "Number of LiDAR frames and images must be the same"

# Set up the figure with two subplots: one for the LiDAR data (polar plot) and one for the camera images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Set up the LiDAR plot in polar coordinates
ax1 = plt.subplot(121, projection='polar')  # Create a polar subplot
angles = np.deg2rad(np.linspace(0, 359, 180))  # Generate angles from 0 to 359 degrees with 2-degree increments
line, = ax1.plot(angles, lidar_data[0])  # Initial frame
ax1.set_title("LiDAR Data Visualization")
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)

# Set up the camera image plot
img_display = ax2.imshow(images[0])
ax2.axis('off')  # Hide the axis
ax2.set_title("Camera Image")

# Update function for animation
def update(frame):
    # Update LiDAR plot
    line.set_ydata(lidar_data[frame])
    
    # Update camera image plot
    img_display.set_array(images[frame])
    
    return line, img_display

# Create an animation
ani = FuncAnimation(fig, update, frames=len(lidar_data), interval=100, blit=True)

plt.show()

# # Set up the figure and polar axis
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.set_title("Animated LiDAR Data Visualization")
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)

# # Initial plot setup
# angles = np.deg2rad(np.linspace(0, 359, 180))
# line, = ax.plot(angles, lidar_data[0])  # Initial frame

# def update(frame):
#     line.set_ydata(lidar_data[frame])  # Update the data of the plot
#     return line,

# # Create an animation
# ani = FuncAnimation(fig, update, frames=len(lidar_data), interval=100, blit=True)

# plt.show()