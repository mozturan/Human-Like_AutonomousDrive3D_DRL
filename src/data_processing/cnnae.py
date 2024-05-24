from keras.layers import (Conv2D, MaxPooling2D, 
                          Reshape, UpSampling2D, 
                          Dense, MaxPool2D, Flatten,
                          Conv2DTranspose)
import keras
from keras.optimizers.legacy import Adam

class ConvolutionalAutoencoder:
    def __init__(self, input_shape=(80, 160)):
        self.input_shape = input_shape
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = keras.Input(shape=self.input_shape)
        x = input_img
        #expand dims
        x = keras.layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(x)

        #Encoder
        x = Conv2D(32, 8, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, 4, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pool_size=2, strides=1, name="MaxPool2D")(x)        # x = Flatten(name = "Flatten",data_format='channels_last')(x)

        # x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(encoded)
        x = Conv2DTranspose(64, 4, strides=2, activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, 8, strides=2, activation='relu', padding='same')(x)
        decoded = Conv2DTranspose(1, 8, strides=2, activation='sigmoid', padding='same')(x)
        
        autoencoder = keras.Model(input_img, decoded)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        return autoencoder    
    
    def train(self, X_train, X_test, epochs=10, batch_size=64):
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
        encoder = keras.Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer(index=-2).output)
        encoder_json = encoder.to_json()
        with open(encoder_file, "w") as json_file:
            json_file.write(encoder_json)
        encoder.save_weights(weights_file)

    def backup(self):

        #Encoder
        x = Conv2D(32, 4, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, 4, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, strides=1, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pool_size=2, strides=1, name="MaxPool2D")(x)        # x = Flatten(name = "Flatten",data_format='channels_last')(x)

        # Decoder
        x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(encoded)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(32, (4,4), strides=2, activation='relu', padding='same')(x)
        x = UpSampling2D(size=(4,4))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        decoded = UpSampling2D(size=(2,2))(decoded)

        