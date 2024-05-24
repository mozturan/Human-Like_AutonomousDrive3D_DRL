from keras.layers import (Conv2D, MaxPooling2D, 
                          Reshape, UpSampling2D, 
                          Dense, MaxPool2D, Flatten,
                          Conv2DTranspose)
import keras
from keras.optimizers.legacy import Adam
import numpy as np

class ConvolutionalAutoencoder:
    def __init__(self, input_shape=(120, 160)):
        self.input_shape = input_shape
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = keras.Input(shape=self.input_shape)
        x = input_img
        #expand dims
        # x = keras.layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(x)

        # Encoder
        conv1 = Conv2D(16, kernel_size=4, strides=2, activation='relu', padding='same')(x)
        conv2 = Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='same')(conv1)
        conv3 = Conv2D(64, kernel_size=4, strides=2, activation='relu', padding='same')(conv2)
        conv4 = Conv2D(128, kernel_size=4, strides=2, activation='relu', padding='same')(conv3)

        # Flatten
        flatten = Flatten()(conv4)

        # Compute the shape before flatten
        self.shape_before_flatten = conv4.shape[1:]

        # Encoder linear layer
        encode_linear = Dense(self.z_size, name='encoder_linear')(flatten)

        # Decoder linear layer
        decode_linear = Dense(int(np.prod(self.shape_before_flatten)))(encode_linear)

        # Reshape
        reshape = Reshape(self.shape_before_flatten)(decode_linear)

        # Decoder
        deconv1 = Conv2DTranspose(64, kernel_size=4, strides=2, activation='relu', padding='same')(reshape)
        deconv2 = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', padding='same')(deconv1)
        deconv3 = Conv2DTranspose(16, kernel_size=5, strides=2, activation='relu', padding='same')(deconv2)
        decoded = Conv2DTranspose(self.input_shape[2], kernel_size=4, strides=2, activation='sigmoid', padding='same')(deconv3)


        autoencoder = keras.Model(input_img, decoded)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        return autoencoder    
    
    def train(self, X_train, X_test, epochs=100, batch_size=64):
        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test))

    def predict(self, X):
        
        # Predicts reconstruction for given images
        
        return self.autoencoder.predict(X)

    
    def summary(self):
        """
        Prints a summary of the Autoencoder's architecture
        """
        print("Autoencoder Architecture:")
        self.autoencoder.summary()

    def save(self, encoder_file="encoder_model.json", weights_file="encoder_weights.h5"):
        encoder = keras.Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer(index=-7).output)
        encoder_json = encoder.to_json()
        with open(encoder_file, "w") as json_file:
            json_file.write(encoder_json)
        encoder.save_weights(weights_file)


