import tensorflow as tf
from tensorflow import keras

MODELNAME = "alexnet"


# alexnet
# define model as list of keras layers
model = keras.Sequential([
    keras.layers.Conv2D(96, (11,11),  strides = 4, padding = "same", activation = "relu", input_shape = (244,244,4)),
    # keras.layers.Lambda(tf.nn.local_response_normalization),
    keras.layers.MaxPooling2D((3, 3), strides=2),

    keras.layers.Conv2D(256, (5,5), strides = 1, padding = "same", activation = "relu"),
    # keras.layers.Lambda(tf.nn.local_response_normalization),
    keras.layers.MaxPooling2D((3, 3), strides=2),

    keras.layers.Conv2D(384, (3,3), strides = 1, padding = "same", activation = "relu"),

    keras.layers.Conv2D(384, (3,3), strides = 1, padding='same', activation="relu"),

    keras.layers.Conv2D(256, (3,3), strides = 1, padding='same', activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=2),

    keras.layers.Flatten(),

    keras.layers.Dense(4096, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(5, activation = "relu")
    ])









])

model.summary()

model.save('untrained/' + MODELNAME)