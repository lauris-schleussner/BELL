from bellutils.gradcam import GradCAM
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


imgpath = "E:/BELL/resized/184897.jpg"
modelpath = "E:/BELL/models/05_10_2022_15_52_25/saved_vgg16"


image = tf.io.read_file(imgpath)
image = tf.image.decode_jpeg(image, channels=3)
image /= 255 # normalize to [0,1] range
image = np.expand_dims(image, axis=0)


# load model
model = tf.keras.models.load_model(modelpath)


# get layer names
for idx in range(len(model.layers)):
  print(model.get_layer(index = idx).name)


# predict
preds = model.predict(image) 
i = np.argmax(preds[0])

# load gradcam class
icam = GradCAM(model, i)
heatmap = icam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (244, 244))

image = cv2.imread(imgpath)
image = cv2.resize(image, (244, 244))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

fig, ax = plt.subplots(1, 3)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)

plt.show()