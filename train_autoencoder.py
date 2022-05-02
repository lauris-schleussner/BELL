# Autoencoder will train with ALL images!

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Dense, Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
from tqdm import tqdm
import pickle
import os

# https://www.analyticsvidhya.com/blog/2021/01/querying-similar-images-with-tensorflow/

def main(EPOCHS):

  # Load images
  img_height = 244
  img_width = 244
  channels = 3
  batch_size = 16

  train_datagen = ImageDataGenerator(rescale=1./255,
                                    validation_split=0.2)

  training_set = train_datagen.flow_from_directory(
      './autoenc',
      target_size = (img_height, img_width),
      batch_size = batch_size,
      class_mode = 'input',
      subset = 'training',
      shuffle=True)

  validation_set = train_datagen.flow_from_directory(
      './autoenc',
      target_size = (img_height, img_width),
      batch_size = batch_size,
      class_mode = 'input',
      subset = 'validation',
      shuffle=False)

  if not os.path.isdir("autoenc/resized"):
    raise ValueError("No folder found, was the 'resized' folder copied to 'autoenc/resized'?")

  # Define the autoencoder
  input_model = Input(shape=(img_height, img_width, channels))

  # Encoder layers
  encoder = Conv2D(32, (3,3), padding='same', kernel_initializer='normal')(input_model)
  encoder = LeakyReLU()(encoder)
  encoder = BatchNormalization(axis=-1)(encoder)

  encoder = Conv2D(64, (3,3), padding='same', kernel_initializer='normal')(encoder)
  encoder = LeakyReLU()(encoder)
  encoder = BatchNormalization(axis=-1)(encoder)

  encoder = Conv2D(64, (3,3), padding='same', kernel_initializer='normal')(input_model)
  encoder = LeakyReLU()(encoder)
  encoder = BatchNormalization(axis=-1)(encoder)

  encoder_dim = K.int_shape(encoder)
  encoder = Flatten()(encoder)

  # Latent Space
  latent_space = Dense(16, name='latent_space')(encoder)

  # Decoder Layers
  decoder = Dense(np.prod(encoder_dim[1:]))(latent_space)
  decoder = Reshape((encoder_dim[1], encoder_dim[2], encoder_dim[3]))(decoder)

  decoder = Conv2DTranspose(64, (3,3), padding='same', kernel_initializer='normal')(decoder)
  decoder = LeakyReLU()(decoder)
  decoder = BatchNormalization(axis=-1)(decoder)

  decoder = Conv2DTranspose(64, (3,3), padding='same', kernel_initializer='normal')(decoder)
  decoder = LeakyReLU()(decoder)
  decoder = BatchNormalization(axis=-1)(decoder)

  decoder = Conv2DTranspose(32, (3,3), padding='same', kernel_initializer='normal')(decoder)
  decoder = LeakyReLU()(decoder)
  decoder = BatchNormalization(axis=-1)(decoder)

  decoder = Conv2DTranspose(3, (3, 3), padding="same")(decoder)
  output = Activation('sigmoid', name='decoder')(decoder)

  # Create model object
  autoencoder = Model(input_model, output, name='autoencoder')

  # Compile the model
  autoencoder.compile(loss="mse", optimizer= Adam(learning_rate=1e-3))

  # Fit the model
  history = autoencoder.fit_generator(
            training_set,
            steps_per_epoch=training_set.n // batch_size,
            epochs = EPOCHS,
            validation_data=validation_set,
            validation_steps=validation_set.n // batch_size,
            callbacks = [ModelCheckpoint('autoenc/models/image_autoencoder_2.h5', 
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=False)])

  autoencoder.summary()

  latent_space_model = Model(
                        autoencoder.input, 
                        autoencoder.get_layer('latent_space').output)

  # Load all images and predict them with the latent space model
  X = []
  indices = []

  for i in tqdm(range(len(os.listdir('./autoenc/resized')))):
    try:
      img_name = os.listdir('./autoenc/resized')[i]
      img = load_img('./autoenc/resized/{}'.format(img_name), 
                    target_size = (244, 244))
      img = img_to_array(img) / 255.0
      img = np.expand_dims(img, axis=0)
      pred = latent_space_model.predict(img)
      pred = np.resize(pred, (16))
      X.append(pred)
      indices.append(img_name)

    except Exception as e:
      print(img_name)
      print(e)

      # Export the embeddings
  embeddings = {'indices': indices, 'features': np.array(X)}
  pickle.dump(embeddings, 
              open('./autoenc/image_embeddings.pickle', 'wb'))


if __name__ == "__main__":
  main(2)