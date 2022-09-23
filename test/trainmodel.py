# train a test model cats/dogs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "E:/BELL/test/training_set",
  validation_split=0.2,
  subset="training",
  seed=123,
  batch_size=32,
  label_mode='categorical')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "E:/BELL/test/training_set",
  validation_split=0.2,
  subset="validation",
  seed=123,
  batch_size=32,
  label_mode='categorical')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    layers.experimental.preprocessing.Resizing(100, 100),
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(100, 100, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.fit(train_ds, validation_data=val_ds, epochs = 1, shuffle = True)
model.summary()


model.save("cat_dog_model")