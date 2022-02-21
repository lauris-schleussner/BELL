# classify style by image size proof of concept

import tensorflow as tf
import sqlite3
from tqdm import tqdm
import numpy as np
from pymage_size import get_image_size
import tensorflow as tf
import datetime


 # create connection to database
conn = sqlite3.connect("database.db")
c = conn.cursor()

# get list of all unique styles returns list of tupels with one datapair each
res = c.execute("SELECT DISTINCT style FROM artworks WHERE used = True").fetchall()
classlist = []
for i in res:
        classlist.append(i[0])


res = c.execute("SELECT filename, style FROM artworks WHERE used = True ORDER BY RANDOM() LIMIT 7517").fetchall()
stylelist = []
dimensionlist = []
for filename, style in tqdm(res):
    img_format = get_image_size("originalsize/" + filename)
    w, h = img_format.get_dimensions()

    # height as bigger dim
    if w >= h:
        w,h = w,h

    w = w/2000 
    h = h/2000

    idx = classlist.index(style)
    label = [0]*len(classlist)
    label[idx] = 1

    stylelist.append(label)
    dimensionlist.append([w,h])


trainstyle_ds = np.asarray(stylelist[0:6014])
teststyle_ds = np.asarray(stylelist[6014:7517])


traindim_ds = np.asarray(dimensionlist[0:6014])
testdim_ds = np.asarray(dimensionlist[6014:7517])

print(len(traindim_ds))
print(len(testdim_ds))
print(len(trainstyle_ds))
print(len(teststyle_ds))

'''
# model
inputs = tf.keras.Input(shape=(2,))
dense = tf.keras.layers.Dense(64, activation="relu")
x = dense(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

'''

model = tf.keras.Sequential([
    
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation="relu") # classnumber = 5
    ])

model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# history = model.fit(traindim_ds, trainstyle_ds, batch_size=128, epochs=100, validation_split=0.2, callbacks=[tensorboard_callback])   
history = model.fit(traindim_ds, trainstyle_ds, batch_size=512, epochs=1000, validation_split=0.2) 

print(testdim_ds)
print(teststyle_ds)

test_scores = model.evaluate(testdim_ds, teststyle_ds, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
