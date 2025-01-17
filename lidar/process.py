from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_data(data_dir):

    originals = []
    images = []

    for filename in os.listdir(data_dir)[:-1]:
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path, cv2.COLOR_RGB2BGR)
        originals.append(img)

        img = np.array(img)
        img = crop_image(img)
        img = preprocess_image(img)
        images.append(img)

    dataset = np.array(images)

    return dataset, originals

def crop_image(image):
        return image[40:120, :, :]
        
def preprocess_image(image):
        """
        Preprocess the image by normalizing it and converting it to uint8
        """
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image / 255.0
 
def prepare_data(images):
    X = images.copy()
    y = to_categorical(np.arange(images.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=False)

    return X_train, X_test, y_train, y_test

def visualize_samples(autoencoder, X_test, test_samples):
    preds = [autoencoder.predict(np.expand_dims(X_test[i], axis=0)) for i in test_samples]

    for i, sample in enumerate(test_samples):
        img = X_test[sample]
        pred = preds[i]

        #add axis to img
        img = np.expand_dims(img, axis=0)
        plt.imshow(img[0])
        plt.show()

        plt.imshow(pred[0])
        plt.show()

        #save img and pred
        plt.imsave(f"sample{i}_epoch100.png", img[0])
        plt.imsave(f"pred{i}_epoch100.png", pred[0])

    plt.show()


def rgb_to_grayscale(images):
    """
    Convert color images to grayscale
    """
    grayscale_images = []
    for image in images:
        grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        grayscale_images.append(grayscale_image)

    return np.array(grayscale_images)

def blur_images(images, kernel_size=15):
    """
    Apply Gaussian blur to images
    """
    blurred_images = []
    for image in images:
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        blurred_images.append(blurred_image)

    return np.array(blurred_images)