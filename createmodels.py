import tensorflow as tf

MODELNAME = "testmodel2"

# define model as list of keras layers
model = tf.keras.Sequential([
    # rescaling layer
    tf.keras.layers.Rescaling(1./255, input_shape=(244, 244, 3)),

    # layer 1
    tf.keras.layers.Conv2D(2, (3,3), padding='valid', activation="relu",),
    tf.keras.layers.MaxPooling2D((8, 8), strides=4),

    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(5, activation="relu") # classnumber = 5
    ])

model.summary()

# configure model for training
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Display the model's architecture
model.summary()

model.save('untrained/' + MODELNAME)