from src.data_processing.cnnae import ConvolutionalAutoencoder
from src.data_processing.process import *

if __name__ == '__main__':

    data_dir = '/home/o/Documents/donkeycar_rl/data/human'
    images, originals = load_data(data_dir)

    X_train, X_test, y_train, y_test = prepare_data(images)
    autoencoder = ConvolutionalAutoencoder(input_shape=(80, 160, 3), z_size=32)
    autoencoder.train(X_train, X_test, epochs=50, batch_size=32)

    visualize_samples(autoencoder, X_test, [100,500,1000,1500])

    autoencoder.save_encoder()
    autoencoder.save_ae()