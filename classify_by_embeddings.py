import tensorflow as tf
import pickle
import numpy as np

# encodings = pickle.load(open("img_embed_xception_after_umap.pickle", "rb" ))
encodings = pickle.load(open("image_embeddings_16_ep.pickle", "rb" ))

stylelist = encodings["indices"]
points_from_file = encodings["features"]

points = []
for x, y in zip(points_from_file[0], points_from_file[1]):
    points.append([x,y])

idx_list = []
# get unique styles (all 5)
uniquestyles = list(set(stylelist))

for style in stylelist:

    idx = uniquestyles.index(style)
    label = [0]*len(uniquestyles)
    label[idx] = 1

    idx_list.append(label)


trainstyle_ds = np.asarray(idx_list)
traindim_ds = np.asarray(points)

print(len(traindim_ds))
print(len(trainstyle_ds))

model = tf.keras.Sequential([
    
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(5, activation="relu") # classnumber = 5
    ])

model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)

history = model.fit(traindim_ds, trainstyle_ds, batch_size=16, epochs=50, validation_split=0.2) 

##########################################################
# plot
#############################################################
from matplotlib import pyplot as plt

# build one batch with all test pictures 
# Final tensor format (Number of Pictures, IMGSIZE, IMGSIZE, 3)

import pandas as pd
# plot history
dafr = pd.DataFrame(history.history)
print(dafr)
dafr = dafr.drop(columns = ["accuracy", "loss", "val_loss"])

dafr.plot(figsize=(8,5))


plt.grid(True)
plt.gca().set_ylim(0.2, 0.6)
plt.savefig("classify_by_embeddings" + '_history.png', dpi=500)
print("saved plotting results for" + "2D dmbedding classification")