import tensorflow as tf
import pickle as pkl
import numpy as np
from bellutils.getfrompath import getfrompath
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import sqlite3

# db
conn = sqlite3.connect("database.db")
c = conn.cursor()

res = c.execute("SELECT filename, style FROM artworks WHERE used = True and corrupt = False").fetchall()

all_filenames = []
for filename, style in res:
    all_filenames.append([filename, style])


modelpath = "models/saved_xception"
model = tf.keras.models.load_model(modelpath)

# printa all layers
for idx in range(len(model.layers)):
  print(model.get_layer(index = idx).name)

all_filenames = all_filenames[:10000]
print(len(all_filenames))

latent_space = tf.keras.models.Model(model.input, model.get_layer("avg_pool").output)


predictions = []
indices = []
for path, style in tqdm(all_filenames):

    try:

        img = load_img("resized/" + path, target_size = (244, 244))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = latent_space.predict(img)[0]

        # pred = np.resize(pred, (16))

        predictions.append(pred)
        indices.append([path, style])
    except:
        pass

embeddings = {'indices': indices, 'features': np.array(predictions)}
pkl.dump(embeddings, open('img_embed_xception_actuallynonpretrained.pickle', 'wb'))
