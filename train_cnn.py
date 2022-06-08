# train default cnn 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
import sqlite3
import tensorflow_datasets as tfds
import random
import numpy
import wandb
# wandb.init(project="BELL", save_code=False)
# wandb.init(project="bell", entity="lauris_bell", tags=['cnn'])

from wandb.keras import WandbCallback
from datetime import datetime


# getting the dataset from the database has been outsourced
from bellutils.get_datasets import get_datasets

# network parameter settings
BATCHSIZE = 32
LEARNINGRATE = 0.001 # default Adam learning rate
IMGSIZE = 100 # 244 # images are rescaled to a square, size in px

# paths
DBNAME = "database.db"
SAVEPATH = "models/" + datetime.now().strftime('%m_%d_%Y_%H_%M_%S')+ "/"
MODELPATH = SAVEPATH + "saved_cnn/"
# CPPATH = SAVEPATH + "cnn_checkpoint.ckpt"


# database
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

# Tensorflow Stuff 
AUTOTUNE = tf.data.AUTOTUNE

# open and read image. This function is mapped to every dataset row
def preprocess_image(filename, label):

    image = tf.io.read_file("resized/" + filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255 # normalize to [0,1] range

    return image, label

def main(EPOCHS):

    run_tags = ['cnn']
    wandb.init(project="bell", entity="lauris_bell", tags=run_tags)

    # get dataset
    train_ds = get_datasets("train")
    val_ds = get_datasets("validation")
    test_ds = get_datasets("test")


    # shuffle datasets
    train_ds = train_ds.shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)
    val_ds = val_ds.shuffle(val_ds.cardinality(), reshuffle_each_iteration=True)
    test_ds = test_ds.shuffle(test_ds.cardinality(), reshuffle_each_iteration=True)

    # load and preprocess images
    train_ds = train_ds.map(preprocess_image)
    val_ds = val_ds.map(preprocess_image)
    test_ds = test_ds.map(preprocess_image)

    # info
    print("train_ds", train_ds.cardinality())
    print("val_ds", val_ds.cardinality())
    print("test_ds", test_ds.cardinality())

    train_ds = train_ds.batch(BATCHSIZE)
    val_ds = val_ds.batch(BATCHSIZE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    
    # load model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 5, activation="relu", padding='same', input_shape = (IMGSIZE, IMGSIZE, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(5, activation='softmax'),
    ])

    # "Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses."
    # gradient descent to improve training
    optimizer = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

    # configure model for training
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # callbacks that are triggered during training, create checkpoints
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CPPATH, save_weights_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        # callbacks=[cp_callback, es_callback, WandbCallback(save_model = False)],
        callbacks=[es_callback, WandbCallback(save_model = False)]
    )

    # after sucessfull run save model
    model.save(MODELPATH)

    wandb.finish()

    print("succesfully trained")

    return [model, history, test_ds]

if __name__ == "__main__":
    main(EPOCHS=50)
