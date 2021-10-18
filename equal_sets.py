# https://www.tensorflow.org/tutorials/load_data/images

import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import PIL.Image
import tensorflow as tf
import glob
import random
from termcolor import cprint
import sys
import tensorboard
import datetime

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

checkpoint_path = "models/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

class_names = os.listdir('D:/BELL/IADataset')

filelist = []
filesPerClass = 1000
classNames2Remove = []

# take same amount of images from each class
for classname in class_names:

  classlist = glob.glob("D:/BELL/IADataset/" + classname + "/*.jpg")
  random.shuffle(classlist)

  if len(classlist) >= filesPerClass:
    filelist += classlist[0:filesPerClass]

  else:
    # print(classname , "doesnt contain enough images:", len(classlist))
    classNames2Remove.append(classname)

for nameRemove in classNames2Remove:
  class_names.remove(nameRemove)
  # print("removed class", nameRemove)

class_count = len(class_names)
image_count = len(filelist)
print("total classes after selection", len(class_names))

print("file total:", len(filelist))

class_names = np.sort(np.array(class_names))
print(class_names)

# shuffle list of paths
list_ds = tf.data.Dataset.list_files(filelist)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  img_height = 500
  img_width = 500
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(class_names.size)
])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=50,
  callbacks=[tensorboard_callback, cp_callback]
)

keras_model_path = "D:/BELL/models"
model.save(keras_model_path)