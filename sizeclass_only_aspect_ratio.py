# classify style by image size proof of concept

import tensorflow as tf
import sqlite3
from tqdm import tqdm
import numpy as np
from pymage_size import get_image_size
import tensorflow as tf
import datetime
import numpy


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

fulllist = []
for filename, style in res:
    fulllist.append([filename, style])

numpy.random.shuffle(fulllist)

for filename, style in tqdm(fulllist):
    img_format = get_image_size("originalsize/" + filename)
    w, h = img_format.get_dimensions()

    w = w/2000 
    h = h/2000

    # calculate aspect ratio
    ar = min([w,h]) / max([w,h])


    idx = classlist.index(style)
    label = [0]*len(classlist)
    label[idx] = 1

    stylelist.append(label)
    dimensionlist.append(ar)


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
    
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# history = model.fit(traindim_ds, trainstyle_ds, batch_size=128, epochs=100, validation_split=0.2, callbacks=[tensorboard_callback])   
history = model.fit(traindim_ds, trainstyle_ds, batch_size=16, epochs=600, validation_split=0.2) 


##########################################################
# plot
#############################################################
from matplotlib import pyplot as plt

# build one batch with all test pictures 
# Final tensor format (Number of Pictures, IMGSIZE, IMGSIZE, 3)
'''
images = np.array(images)
labels = np.array(labels)
labels = labels.flatten()
'''
# print("labels: ", labels)

teststyle_ds = teststyle_ds.tolist()

# convert encoded version back to style index
labellist = []
for encoded in teststyle_ds:
    idx = encoded.index(1)
    #label = classlist[idx] 
    labellist.append(idx)

rawpredictions = model.predict(testdim_ds)

argmaxpredictions = tf.argmax(rawpredictions, axis=-1)
argmaxpredictions = argmaxpredictions.numpy()
argmaxpredictions = argmaxpredictions.flatten()
# print("argmaxpredictions:", argmaxpredictions)

weights = []

for i in range(len(rawpredictions)):
    rawsingleprediction = rawpredictions[i]
    argmaxindex = argmaxpredictions[i]
    weights.append(rawsingleprediction[argmaxindex])


confusion = tf.math.confusion_matrix(labels=labellist, predictions=argmaxpredictions)
# print(confusion.shape)

conf_matrix = confusion.numpy()
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plottitle = "sizeclass_only_ac"
outfolder = "E:/BELL/plots/"

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
# plt.show()
plt.savefig(outfolder + plottitle + '_cm.png', dpi=500)

import pandas as pd
# plot history
dafr = pd.DataFrame(history.history)
print(dafr)
dafr = dafr.drop(columns = ["accuracy", "loss", "val_loss"])

dafr.plot(figsize=(8,5))


plt.grid(True)
plt.gca().set_ylim(0.2, 0.35)
plt.savefig(outfolder + plottitle + '_history.png', dpi=500)
print("saved plotting results for", plottitle)
