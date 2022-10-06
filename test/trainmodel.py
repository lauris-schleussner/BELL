# train a test model cats/dogs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cfm_history


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

print(val_ds)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
"""
model = Sequential([
    layers.experimental.preprocessing.Resizing(100, 100, input_shape = (256,256,3)),
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(100, 100, 3)),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])
"""
def processinput(image, label):

    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label

val_ds.map(processinput)
train_ds.map(processinput)

# original model
core  = tf.keras.applications.xception.Xception(include_top=False, weights="imagenet", input_shape= (256, 256, 3), pooling=max)
core.trainable = False


x = tf.keras.layers.GlobalAveragePooling2D()(core.output)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
end = tf.keras.layers.Dense(2)(x)

model = tf.keras.Model(inputs = core.inputs, outputs = end)

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights = True)

history = model.fit(train_ds, validation_data=val_ds, epochs = 5, shuffle = True, callbacks = [callback])

traindata = [model, history, val_ds]

cfm_history.main(traindata, "cat_dog_plot")

model.save("E:/BELL/cat_dog_model/cat_dog_model_besser_xception_larger")