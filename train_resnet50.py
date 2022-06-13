# TODO implement SSIM, PSNR

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
import sqlite3
import wandb
# wandb.init(project="BELL", save_code=False)
# wandb.init(project="bell", entity="lauris_bell", tags=['resnet50'])

import atexit

from wandb.keras import WandbCallback
from datetime import datetime

# getting the dataset from the database has been outsourced
from bellutils.get_datasets import get_datasets

# network parameter settings
BATCHSIZE = 32
LEARNINGRATE = 0.0001 # 0.001 # default Adam learning rate
IMGSIZE = 244 # 100 # 244 # images are rescaled to a square, size in px

# paths
DBNAME = "database.db"
SAVEPATH = "models/" + datetime.now().strftime('%m_%d_%Y_%H_%M_%S')+ "/"
# MODELPATH = SAVEPATH + "saved_resnet50/"
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
    # image /= 255 # normalize to [0,1] range

    # special resnet preprocessing
    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label

def main(EPOCHS, WAB_FLAG, pretrained, add_tags=list(), savemodel=True):

    if WAB_FLAG:
        run_tags = ['resnet50']
        if pretrained:
            run_tags.append('pretrained')
        wandb.init(project="bell", entity="lauris_bell", tags=run_tags+add_tags)

    if pretrained:
        MODELPATH = SAVEPATH + "saved_resnet50_finetuned/"
    else:
        MODELPATH = SAVEPATH + "saved_resnet50/"

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

    train_ds = train_ds.batch(BATCHSIZE)
    val_ds = val_ds.batch(BATCHSIZE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():

        if pretrained:
            pretrained_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape= (IMGSIZE, IMGSIZE, 3), pooling=max)
            pretrained_model.trainable = False

            x = tf.keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

            # add dropout and 3 dense layers ontop
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(1024,activation='relu')(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

            model = tf.keras.Model(pretrained_model.inputs, outputs)

        else:
            model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape= (IMGSIZE, IMGSIZE, 3), pooling=max, classes = 5)


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

    if WAB_FLAG:
        callbacks = [es_callback, WandbCallback(save_model = False)]
    else:
        callbacks = [es_callback]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        # callbacks=[cp_callback, es_callback, WandbCallback(save_model = False)]
        # callbacks=[es_callback, WandbCallback(save_model = False)]
        callbacks=callbacks
    )

    if savemodel:
        # after sucessfull run save model
        model.save(MODELPATH)

    if WAB_FLAG:
        wandb.finish()
    
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    return [model, history, test_ds]

if __name__ == "__main__":
    # main(EPOCHS=3, WAB_FLAG=True, pretrained=True, add_tags=['testrun'], savemodel=False)
    main(EPOCHS=3, WAB_FLAG=False, pretrained=False, savemodel=False)
