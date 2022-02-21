# TODO implement SSIM, PSNR


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf # tensorboard crashes python
from tensorflow import keras
import sqlite3
import numpy as np
# import wandb as wb
# from wandb.keras import WandbCallback

def main():
    # network parameter settings
    BATCHSIZE = 16
    EPOCHS = 10
    # paths
    MODELNAME = "testmodel2"

    UNTRAINEDMODELPATH = "untrained/"
    TRAINEDMODELPATH = "trained/"
    CPOINTPATH = "cpoint/"

    DBNAME = "database.db"

    # Tensorflow Stuff 
    AUTOTUNE = tf.data.AUTOTUNE

    # create connection to database
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    # get list of all unique styles returns list of tupels with one datapair each
    res = c.execute("SELECT DISTINCT style FROM artworks WHERE used = True").fetchall()
    classlist = []
    for i in res:
            classlist.append(i[0])

    print("training with classes:", classlist)

    # select training ds
    querry = "SELECT filename, style FROM artworks WHERE used = True AND train = True ORDER BY RANDOM() LIMIT 160"
    c.execute(querry)
    res = c.fetchall()
    train_ds = tf.data.Dataset.from_tensor_slices(res)
    train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True)

    # select val ds
    querry = "SELECT filename, style FROM artworks WHERE used = True AND validation = True ORDER BY RANDOM() LIMIT 160 "
    c.execute(querry)
    res = c.fetchall()
    val_ds = tf.data.Dataset.from_tensor_slices(res)
    val_ds = val_ds.shuffle(len(val_ds), reshuffle_each_iteration=True)

    print("train_ds", train_ds.cardinality())
    print("val_ds", val_ds.cardinality())

    # for every datapair [filename, label]:
    @tf.function
    def processImage(datapair):

        one_hot = datapair[1] == classlist
        label = tf.argmax(one_hot)
       
        # prescaled images are used
        path = "resized/" + datapair[0]

        img = tf.io.read_file(path) # Load the raw data from the file as a string
        img = tf.io.decode_image(img, channels=3, expand_animations = False)
        img = tf.cast(img, tf.float32)
        
        return img, label
        

    # data processing function is mapped to each [path, label combo]
    train_ds = train_ds.map(processImage, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(processImage, num_parallel_calls=AUTOTUNE)

    # configure for performance # TODO look into this and optimise further
    def configure_for_performance(ds):
        ds = ds.cache()
        # ds = ds.shuffle(buffer_size=1000) TODO why shuffle again?

        # batch and fetch the next batch after previous one is finished
        ds = ds.batch(BATCHSIZE, drop_remainder = True)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # performace function is mapped to each ds
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    # load model
    model = tf.keras.models.load_model(UNTRAINEDMODELPATH + MODELNAME)

    # checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CPOINTPATH + MODELNAME + "/cpoints", save_weights_only=True, verbose=1)

    model.summary()

    model.fit(
        train_ds,        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[cp_callback]
    )

    # after sucessfull run save model
    model.save(TRAINEDMODELPATH + MODELNAME)
    print("model saved at", TRAINEDMODELPATH + MODELNAME)

if __name__ == "__main__":
    main()
