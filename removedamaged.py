import tensorflow as tf
import os
from PIL import Image
import sqlite3

tf.executing_eagerly()

conn = sqlite3.connect("WikiartDataset.db")
c = conn.cursor()

c.execute("SELECT path FROM artworks")
res = c.fetchall()

c = 0
for path in res:
    c += 1
    if c % 1000 == 0:
        print(c)

    try:
        img = tf.io.read_file(path[0]) # Load the raw data from the file as a string
        img = tf.io.decode_image(img, channels=3, expand_animations = False) # decode image
        img = tf.image.resize(img, [200, 200]) # resize image
    except:
        os.remove(path)
        print("removed", path)