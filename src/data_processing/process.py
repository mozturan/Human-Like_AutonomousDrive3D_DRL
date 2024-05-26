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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=True)

    return X_train, X_test, y_train, y_test

def visualize_samples(autoencoder, X_test, test_samples):
    preds = [autoencoder.predict(np.expand_dims(X_test[i], axis=0)) for i in test_samples]

    fig, ax = plt.subplots(len(test_samples), 2)
    for i, sample in enumerate(test_samples):
        img = X_test[sample]
        pred = preds[i]

        ax[i, 0].imshow(img, cmap='gray')
        ax[i, 1].imshow(pred[0], cmap='gray')

    plt.show()
