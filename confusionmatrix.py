# https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model

filepath = "models"

model = load_model(filepath, compile = True)

def decode_img(img):
  img_height = 100
  img_width = 100
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img


input = process_path("D:/BELL/IADataset/Impressionism/abdullah-suriosubroto_air-terjun.jpg")
input = 

print(input.get_shape())
model.summary()
predictions = model.predict(input)
print(predictions)