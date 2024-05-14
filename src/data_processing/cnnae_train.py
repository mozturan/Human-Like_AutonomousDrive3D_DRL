import os
from cnnae import ConvolutionalAutoencoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_data(data_dir):
    """
    Function to load images from a specified directory path, preprocess them, and create a dataset for training.
    Args:
        data_dir (str): The directory path where the images are stored.
    Returns:
        numpy.ndarray: A dataset of preprocessed images for training.
    """
    def load_images_from_path(path):
        images = []
        for filename in os.listdir(path)[:-1]:
            img_path = os.path.join(path, filename)
            # img = load_img(img_path, color_mode="rgb", grayscale=False)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.equalizeHist(img)
            img = img/255.
            # img = rgb2gray(img)
            # img = np.array(img)
            images.append(img)
        return images

    images = load_images_from_path(data_dir)
    dataset = np.array([np.dstack((images[i], images[i+1], images[i+2])) for i in range(len(images)-2)])

    return dataset

def prepare_data(images):
    """
    Prepares the data for training by splitting the input images into training and testing sets.

    Parameters:
    images (numpy.ndarray): The input images to be split.

    Returns:
    numpy.ndarray: X_train - Training data for images.
    numpy.ndarray: X_test - Testing data for images.
    numpy.ndarray: y_train - Training labels for images.
    numpy.ndarray: y_test - Testing labels for images.
    """
    X = images.copy()
    y = to_categorical(np.arange(images.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, shuffle=True)

    return X_train, X_test, y_train, y_test

def merge_datasets(ds1, ds2):
    """
    Merges 2 datasets with same shape to create a one big dataset
    """
    return np.vstack((ds1, ds2))

def visualize_samples(autoencoder, X_test, test_samples):
    """
    Visualizes the original images and their corresponding predictions for a given set of test samples.

    Parameters:
    autoencoder (object): The trained autoencoder model.
    X_test (ndarray): The array of test images.
    test_samples (list): The list of indices of test samples to visualize.

    Returns:
    None
    """
    preds = [autoencoder.predict(np.expand_dims(X_test[i], axis=0)) for i in test_samples]

    fig, ax = plt.subplots(len(test_samples), 2)
    for i, sample in enumerate(test_samples):
        img = X_test[sample]
        pred = preds[i]

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(pred[0])

    plt.show()

if __name__ == '__main__':

    #* we have 2 folders with images
    data_dir1 = '/home/o/Documents/donkeycar_rl/data/images'
    data_dir2 = '/home/o/Documents/donkeycar_rl/data/images2'
    images1 = load_data(data_dir1)
    images2 = load_data(data_dir2)
    images = merge_datasets(images1, images2)

    X_train, X_test, y_train, y_test = prepare_data(images)

    autoencoder = ConvolutionalAutoencoder()
    autoencoder.summary()
    autoencoder.train(X_train, X_test, epochs=10, batch_size=32)

    test_samples = [100, 600, 800, 1000]
    visualize_samples(autoencoder, X_test, test_samples)



