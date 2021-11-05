import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
import prototype

# get data from the other script
model, orderlist, test_ds = prototype.main()

# might work with batching
images, labels = tuple(zip(*test_ds)) 
# images = tf.data.Dataset.from_tensor_slices(images)
# images = images.batch(32, drop_remainder = True)

labels = np.array(labels)
labels = labels.flatten()
print("labels: ", labels)

rawpredictions = model.predict(images)

argmaxpredictions = tf.argmax(rawpredictions, axis=-1)
argmaxpredictions = argmaxpredictions.numpy()
argmaxpredictions = argmaxpredictions.flatten()
print("argmaxpredictions:", argmaxpredictions)

weights = []

for i in range(len(rawpredictions)):
    rawsingleprediction = rawpredictions[i]
    argmaxindex = argmaxpredictions[i]
    weights.append(rawsingleprediction[argmaxindex])

print("weights", weights)

confusion = tf.math.confusion_matrix(labels=labels, predictions=argmaxpredictions, weights = weights)
print(confusion)