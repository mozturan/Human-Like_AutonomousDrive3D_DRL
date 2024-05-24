import os
from cnnae import ConvolutionalAutoencoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

def load_data(data_dir):

    images = []

    for filename in os.listdir(data_dir)[:-1]:
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)

        img = rgb2gray(img)
        img = crop_image(img)
        img = preprocess_image(img)
        img = blur_image(img)
        img = np.expand_dims(img, axis=2)
        images.append(img)

    dataset = np.array(images)

    return dataset

def rgb2gray(rgb):
        """
        Converts an RGB image to grayscale
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def crop_image(image):
        return image[40:120, :]
        
def preprocess_image(image):
        """
        Preprocess the image by normalizing it and converting it to uint8
        """
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image / 255.0
 
def blur_image(image, sigma = 2):
        """
        Blurs the image using a Gaussian Blur
        """        
        return gaussian_filter(image, sigma=2)

def prepare_data(images):
    X = images.copy()
    y = to_categorical(np.arange(images.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, shuffle=True)

    return X_train, X_test, y_train, y_test

def merge_datasets(ds1, ds2):
    return np.vstack((ds1, ds2))

def visualize_samples(autoencoder, X_test, test_samples):
    preds = [autoencoder.predict(np.expand_dims(X_test[i], axis=0)) for i in test_samples]

    fig, ax = plt.subplots(len(test_samples), 2)
    for i, sample in enumerate(test_samples):
        img = X_test[sample]
        pred = preds[i]

        ax[i, 0].imshow(img, cmap='gray')
        ax[i, 1].imshow(pred[0], cmap='gray')

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
    autoencoder.train(X_train, X_test, epochs=15, batch_size=64)

    test_samples = [100, 600, 1000]
    visualize_samples(autoencoder, X_test, test_samples)

    autoencoder.save(encoder_file="models/autoencoder/encoder_model.json", 
                     weights_file="models/autoencoder/encoder_weights.h5")