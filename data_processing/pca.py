import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from process import *

data_dir = '/home/o/Documents/donkeycar_rl/data/generated_track_human'
images, originals = load_data(data_dir)

images = rgb_to_grayscale(images)
images = blur_images(images)

X_train, X_test, y_train, y_test = prepare_data(images)

n_samples, height, width = X_train.shape
flattened_images = X_train.reshape(n_samples, -1)
print(flattened_images.shape)  # Should be (10, 80*160*3)

scaler = StandardScaler()
standardized_images = scaler.fit_transform(flattened_images)

n_components = 100
pca = PCA(n_components=n_components)
pca_images = pca.fit_transform(standardized_images)
print(pca_images.shape)  # Should be (10, 100)

#Reconstruct the original images
reconstructed_images = pca.inverse_transform(pca_images)
reconstructed_images = scaler.inverse_transform(reconstructed_images)
reconstructed_images = reconstructed_images.reshape(n_samples, height, width)
print(reconstructed_images.shape)  # Should be (10, 80, 160, 3)

def plot_images(original, reconstructed, index = 0):

    plt.figure(figsize=(8, 4))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original[index], cmap='gray')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed[index], cmap='gray')
    
    plt.show()

plot_images(X_train, reconstructed_images)

