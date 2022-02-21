import os
import tensorflow as tf
from tensorflow import keras
import sqlite3

# create connection to database
conn = sqlite3.connect("database.db")
c = conn.cursor()

# create model instance
model = tf.keras.Model

# load weights
TRAINEDMODELPATH = "trained/"
MODELNAME = "testmodel2"

model = tf.keras.models.load_model(TRAINEDMODELPATH + MODELNAME)

# get list of all unique styles returns list of tupels with one datapair each
res = c.execute("SELECT DISTINCT style FROM artworks WHERE used = True").fetchall()
classlist = []
for i in res:
        classlist.append(i[0])

# select test ds
querry = "SELECT filename, style FROM artworks WHERE used = True AND test = True ORDER BY RANDOM() LIMIT 160"
resr = c.execute(querry).fetchall()
test_ds = tf.data.Dataset.from_tensor_slices(res)
test_ds = test_ds.shuffle(len(test_ds), reshuffle_each_iteration=True)

def processImage(datapair):

    one_hot = datapair[1] == classlist
    label = tf.argmax(one_hot)
    
    # prescaled images are used
    path = "resized/" + datapair[0]

    img = tf.io.read_file(path) # Load the raw data from the file as a string
    img = tf.io.decode_image(img, channels=3, expand_animations = False)
    img = tf.cast(img, tf.float32)
    
    return img, label
    

# data processing function is mapped to each [path, label combo]
test_ds = test_ds.map(processImage, num_parallel_calls=tf.data.AUTOTUNE)

loss, acc = model.evaluate(test_ds, verbose=2)

print("Restored model, accuracy: {:5.2f}%".format(100 * acc))