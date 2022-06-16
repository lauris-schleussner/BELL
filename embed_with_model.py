import tensorflow as tf
import pickle as pkl
import numpy as np
from bellutils.getfrompath import getfrompath
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

modelpath = "D:/BELL/models/06_13_2022_23_45_33/saved_xception_finetuned"
model = tf.keras.models.load_model(modelpath)

all_images = getfrompath("C:/Users/Lauris/Downloads/resized")
all_images = all_images[:1000]
print(all_images)
print(len(all_images))

latent_space = tf.keras.models.Model(model.input, model.get_layer("global_average_pooling2d_2").output)


predictions = []
indices = []
for path in tqdm(all_images):

    img = load_img(path, target_size = (244, 244))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = latent_space.predict(img)
    pred = np.resize(pred, (16))
    predictions.append(pred)
    indices.append(path)

embeddings = {'indices': indices, 'features': np.array(predictions)}
pkl.dump(embeddings, open('img_embed_xception.pickle', 'wb'))
