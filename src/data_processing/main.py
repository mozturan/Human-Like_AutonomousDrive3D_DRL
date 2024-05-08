import os

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)
        img = load_img(img_path, grayscale=True)
        img = img_to_array(img)
        images.append(img)

    images = np.array(images)
    images = images.reshape(-1, 256, 256, 1)

    return images


def prepare_data(images):
    X = images.copy()
    y = to_categorical(np.arange(images.shape[0]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, epochs=100):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    data_gen = ImageDataGenerator(rescale=1./255)

    train_gen = data_gen.flow_from_directory(
        'data/train',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(train_gen, epochs=epochs, validation_data=X_train, validation_steps=100)


if __name__ == '__main__':
    data_dir = 'data/train'
    images = load_data(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(images)
    train_model(X_train, y_train)
