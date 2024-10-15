from process import *
import keras
import time
import matplotlib.pyplot as plt

def load_encoder(model_folder="models/encoder_tracks/"):
        """
        Loads the encoder model from a json file and the weights from a h5 file
        """
        encoder_file = os.path.join(model_folder) + "encoder_model.json"
        weights_file = os.path.join(model_folder) + "encoder_weights.h5"

        with open(encoder_file, "r") as json_file:
            encoder_json = json_file.read()
        encoder = keras.models.model_from_json(encoder_json)
        encoder.load_weights(weights_file)

        return encoder

def load_ae(model_folder="models/ae_tracks/"):
        """
        Loads the decoder model from a json file and the weights from a h5 file
        """
        decoder_file = os.path.join(model_folder) + "ae_model.json"
        weights_file = os.path.join(model_folder) + "ae_weights.h5"

        with open(decoder_file, "r") as json_file:
            decoder_json = json_file.read()
        decoder = keras.models.model_from_json(decoder_json)
        decoder.load_weights(weights_file)

        return decoder

encoder = load_encoder()
decoder = load_ae()

# '/home/o/Documents/donkeycar_rl/data/test_images'
data_dir = '/home/o/Documents/donkeycar_rl/data/generated_track_human/'
images, originals = load_data(data_dir)

# X_train, X_test, y_train, y_test = prepare_data(images)
test_samples = [100, 500,600,800, 1000, 1500]
decodeds = []
for sample in test_samples:
    xx= np.expand_dims(images[sample], axis=0)
    print(time.time())
    predicted = encoder.predict(xx)
    print(time.time())

    decoded = decoder.predict(xx)
    decodeds.append(decoded)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(images[sample])
    ax[0, 1].imshow(decoded[0])

    plt.show()


# visualize original and decoded

# visualize_samples(decoder, decodeds, [100,500,1000])


