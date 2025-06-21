import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape
import keras
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_shape=(180,), z_size=16):
        self.input_shape = input_shape
        self.z_size = z_size
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input = keras.Input(shape=self.input_shape)
        x = input

        # Encoder
        encoded = Dense(128, activation='relu')(x)
        encoded = Dense(64, activation='relu')(encoded)
        encode_linear = Dense(self.z_size, name='encoder_linear')(encoded)

        # Decoder
        decoded = Dense(64, activation='relu')(encode_linear)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(np.prod(self.input_shape), activation='sigmoid')(decoded)
        decoded_reshape = Reshape(self.input_shape)(decoded)

        autoencoder = keras.Model(input, decoded_reshape)
        autoencoder.compile(optimizer=Adam(0.001), loss='mse')
        # self.encoder = keras.Model(input_img, encode_linear)
        return autoencoder    
    
    def train(self, X_train, X_test, epochs=50, batch_size=64):
        self.autoencoder.fit(X_train, X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_test, X_test),
                             callbacks=[keras.callbacks.EarlyStopping(patience=15, monitor='val_loss')])

    def predict(self, X):
        
        # Predicts reconstruction for given images
        
        return self.autoencoder.predict(X)

    
    def summary(self):
        """
        Prints a summary of the Autoencoder's architecture
        """
        print("Autoencoder Architecture:")
        self.autoencoder.summary()

    def save_encoder(self, model_folder="models/encoder_tracks/"):
        """
        Saves the encoder model to a json file and the weights to a h5 file
        """
        os.makedirs(os.path.dirname(model_folder), exist_ok=True)

        encoder_file = os.path.join(model_folder) + "lidar_encoder.json"
        weights_file = os.path.join(model_folder) + "lidar_encoder.h5"
        #Create the folder if it doesn't exist
        encoder = keras.Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer(index=-5).output)
        encoder_json = encoder.to_json()
        with open(encoder_file, "w") as json_file:
            json_file.write(encoder_json)
        encoder.save_weights(weights_file)


from process import *
from lidar_process import *

path = "/home/o/Documents/donkeycar_rl/lidar/"

lidars = load_multiple_npys(path)
lidars = normalize_lidars(lidars)

X_train, X_test, y_train, y_test = prepare_data(lidars)

ae = Autoencoder()
ae.summary()
ae.train(X_train, X_test, epochs=100, batch_size=16)

ae.save_encoder()
lidar = np.expand_dims(X_train[500
], axis=0)

lidar_predicted = ae.predict(lidar)

# plt.figure(figsize=(10, 10))
# plt.subplot(111, projection='polar')
# angles = np.deg2rad(np.linspace(0, 359, 180))
# line, = plt.plot(angles, lidar[0], label='LİDAR Nokta Bulutu', linewidth=3) 
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(111, projection='polar')
angles = np.deg2rad(np.linspace(0, 359, 180))
line, = plt.plot(angles, lidar[0], label='Original LİDAR Verisi', linewidth=5) 
line1, = plt.plot(angles, lidar_predicted[0], label='Yeniden Oluşturulmus LİDAR Verisi', linewidth=3)
plt.legend(fontsize=23)
plt.show()
