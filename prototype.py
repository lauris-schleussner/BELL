# TODO implement SSIM, PSNR
# TODO drop last batch somehow, as it causes errors with confusionmatrix https://stackoverflow.com/questions/47394731/fixed-size-batches-potentially-discarding-last-batch-using-dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf # tensorboard crashes python
from tensorflow import keras
import sqlite3
# import wandb as wb
# from wandb.keras import WandbCallback



def main():
    # network parameter settings
    BATCHSIZE = 32
    EPOCHS = 1
    LEARNINGRATE = 0.001 # default Adam learning rate
    FULLSIZE = False # True: Images are taken from the "originalsize/" folder and rescaled during execution. False: pre-scaled versions are used. There should not be a difference apart from performance
    IMGSIZE = 200 # images are rescaled to a square, size in px

    # paths
    MODELPATH = "models/"
    checkpoint_path = "models/checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    DBNAME = "database.db"

    # Tensorflow Stuff 
    AUTOTUNE = tf.data.AUTOTUNE

    # Weights and Biases Stuff
    '''
    wb.init(project="BELL", entity="lauris-schleussner")
    wb.config = {
    "learning_rate": LEARNINGRATE,
    "epochs": EPOCHS,
    "batch_size": BATCHSIZE
    }
    '''
    # create connection to database
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    # get list of all unique styles returns list of tupels with one datapair each
    c.execute("SELECT DISTINCT style FROM artworks")
    res = c.fetchall()
    # convert list of tupels to clean list of strings
    classlist = []
    for i in res:
            classlist.append(i[0])
    
    CLASSNUMBER = len(classlist)
    print(CLASSNUMBER)

    # select random path, style tupels from the DB
    querry = "SELECT filename, style FROM artworks ORDER BY RANDOM()"
    c.execute(querry)
    res = c.fetchall()

    IMAGENUMBER = len(res)

    # create dataset from list of all paths
    dataset = tf.data.Dataset.from_tensor_slices(res)

    # shuffle again for good meassure
    dataset = dataset.shuffle(IMAGENUMBER, reshuffle_each_iteration=True)

    # split Dataset 80/10/10 TODO ugly
    val_size = int(IMAGENUMBER * 0.1)
    test_size = int(IMAGENUMBER * 0.1)
    test_ds = dataset.skip(val_size).take(test_size)
    train_ds = dataset.skip(val_size + test_size)
    val_ds = dataset.take(val_size)

    print("fullds", dataset.cardinality())
    print("train_ds", train_ds.cardinality())
    print("val_ds", val_ds.cardinality())
    print("test_ds", test_ds.cardinality())


    # for every datapair [filename, label]:
    def processImage(datapair):

        # argmax encode label: 
        label = datapair[1] == classlist
        label = tf.argmax(label) # argmax encode label

        # prescaled images are used
        if not FULLSIZE: 
            path = "resized/" + datapair[0]

            img = tf.io.read_file(path) # Load the raw data from the file as a string
            img = tf.io.decode_image(img, channels=3, expand_animations = False)
            img = tf.cast(img, tf.float32)
            
            return img, label
        
        # images are rescaled during runtime
        else:
            path = "originalsize/" + datapair[0]

            img = tf.io.read_file(path) # Load the raw data from the file as a string
            img = tf.io.decode_image(img, channels=3, expand_animations = False)
            img = tf.image.resize(img, [IMGSIZE, IMGSIZE])

            return img, label

    # data processing function is mapped to each [path, label combo]
    train_ds = train_ds.map(processImage, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(processImage, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(processImage, num_parallel_calls=AUTOTUNE)

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
    test_ds = configure_for_performance(test_ds)

    # define custom model
    '''
    model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=CLASSNUMBER)
    )
    '''
    # define model as list of keras layers
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu", input_shape = (500, 500, 3)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),

        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu")
        ])

    # "Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses."
    # gradient descent to improve training
    opt = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

    # configure model for training
    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(), # labels are not one-hot encoded # TODO figure out which loss function should be used for which label encoding
        #loss=tf.losses.CategoricalCrossentropy(), # labels are one-hot encoded
        metrics=['accuracy'])

    # callbacks that are triggered during training, create checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # LET THE TRAINING COMMENCE!
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        # callbacks=[cp_callback, WandbCallback()]
        callbacks=[cp_callback]
    )
    # after sucessfull run save model
    model.save(MODELPATH)

    # used by confusionmatrix.py so i dont have to implement dynamically getting labels and paths again...
    return [model, classlist, test_ds]

if __name__ == "__main__":
    # input("to compute confusionmatrix start the other script first, continue regardless? press any key ")
    main()
