from cnnae import ConvolutionalAutoencoder
from process import *

if __name__ == '__main__':

    data_dir = '/home/o/Documents/donkeycar_rl/data/generated_track_human'
    images, originals = load_data(data_dir)

    X_train, X_test, y_train, y_test = prepare_data(images)
    autoencoder = ConvolutionalAutoencoder(input_shape=(80, 160, 3), z_size=32)
    autoencoder.summary()
    autoencoder.train(X_train, X_test, epochs=10, batch_size=16)

    visualize_samples(autoencoder, X_test, [30])

    # autoencoder.save_encoder()
    # autoencoder.save_ae()