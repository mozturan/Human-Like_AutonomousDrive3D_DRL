from process import *
import keras
import time
def load_encoder(model_folder="models/encoder/"):
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

def load_ae(model_folder="models/ae/"):
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
encoder.summary()
time.sleep(5)
decoder = load_ae()
decoder.summary()


data_dir = '/home/o/Documents/donkeycar_rl/data/human'
images, originals = load_data(data_dir)

X_train, X_test, y_train, y_test = prepare_data(images)
print(X_test.shape)
print(X_test[100].shape)
xx= np.expand_dims(X_test[100], axis=0)


predicted = encoder.predict(xx)
print(predicted)


decoded = decoder.predict(xx)
print(decoded)

# visualize original and decoded

visualize_samples(decoder, X_test, [100,500,1000,1500])


