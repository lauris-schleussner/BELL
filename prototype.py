import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import PIL
import PIL.Image
import tensorflow as tf
import glob
import random
import sys
import tensorboard
import datetime
import sqlite3


AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
checkpoint_path = "models/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
DBNAME = "WikiartDataset.db"
CLASSNUMBER = 10

# create connection to dataset
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

c.execute(("SELECT * FROM artworks"))
IMAGENUMBER =len(c.fetchall())

# get list of all unique labels
c.execute("SELECT DISTINCT style FROM artworks")
res = c.fetchall()

# obtain all classnmaes
classnames = []
for i in res:
    for j in list(i):
        try: # error with "NONE"
            jlist = j.split(",")
        except:
            pass
        for k in jlist:
            if k not in classnames:
                classnames.append(k)

# filter out all classes under threshold, takes eternity
orderlist = [] # [genre, number of images]
for genre in classnames:
    c.execute("SELECT COUNT(style) FROM artworks WHERE style LIKE '%" + genre + "%'")
    res = c.fetchone() 
    orderlist.append([genre, res[0]])

# sort result list
orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

# take classes with most results
orderlist = orderlist[0:CLASSNUMBER]

# store the least amount of images
imgPerClass = orderlist[-1][1]

# get list of all classes
classlist = []
for style in orderlist:
    classlist.append(style[0])

# create SQL querry
querry = "SELECT path, style FROM artworks WHERE style IN " + repr(tuple(map(str, classlist))) + " ORDER BY RANDOM() LIMIT " + str(imgPerClass)

# create dataset
dataset = tf.data.experimental.SqlDataset("sqlite", "wikiartDataset.db", "SELECT path, style FROM artworks", (tf.string, tf.string))

# shuffle dataset
dataset = dataset.shuffle(IMAGENUMBER, reshuffle_each_iteration=False)

# split Dataset 80/20
val_size = int(IMAGENUMBER * 0.2)
train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)

def processImage(path, label):
    height = 100
    width = 100

    img = tf.io.read_file(path) # Load the raw data from the file as a string
    print(path)
    img = tf.io.decode_image(img, channels=3, expand_animations = False) # decode image
    img = tf.image.resize(img, [height, width]) # resize image

    label = label == classnames
    label = tf.argmax(label) # argmax encode label

    return img, label

train_ds = train_ds.map(processImage, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(processImage, num_parallel_calls=AUTOTUNE)

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
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(classnames))
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