# TODO implement SSIM, PSNR
# TODO drop last batch somehow, as it causes errors with confusionmatrix https://stackoverflow.com/questions/47394731/fixed-size-batches-potentially-discarding-last-batch-using-dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf # tensorboard crashes python
from tensorflow import keras
import sqlite3
# import wandb as wb
# from wandb.keras import WandbCallback


# network parameter settings
BATCHSIZE = 32
EPOCHS = 100
LEARNINGRATE = 0.001 # default Adam learning rate
FULLSIZE = False # True: Images are taken from the "originalsize/" folder and rescaled during execution. False: pre-scaled versions are used. There should not be a difference apart from performance
IMGSIZE = 100 # images are rescaled to a square, size in px

# paths
MODELPATH = "models/"
checkpoint_path = "models/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
DBNAME = "database.db"

# Tensorflow Stuff 
AUTOTUNE = tf.data.AUTOTUNE


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


def make_data_tuples(res):
    top_labels = ["Impressionism", "Realism", "Romanticism", "Expressionism", "Art_Nouveau_(Modern)"]

    label_names = top_labels
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    
    all_image_paths = list()
    all_label_indexes = list()

    for fname, label in res:
        if (label in top_labels) & (os.path.exists('./resized/'+fname)):
            all_image_paths.append('./resized/'+fname)
            all_label_indexes.append(label_to_index[label])

    # data_tuples = [('./resized/'+fname, label_to_index[label]) for fname, label in res if label in top_labels]
    
    return all_image_paths, all_label_indexes


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMGSIZE, IMGSIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def main():

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

    all_image_paths, all_labels_as_index = make_data_tuples(res)
    # all_labels_as_index = tf.cast(all_labels_as_index, tf.int64)

    # IMAGENUMBER = len(res)
    IMAGENUMBER = len(all_labels_as_index)

    # create dataset from list of all paths
    # dataset = tf.data.Dataset.from_tensor_slices(res)
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels_as_index))

    # image_label_ds = dataset.map(load_and_preprocess_from_path_label)
    dataset = dataset.map(load_and_preprocess_from_path_label)

    # shuffle again for good meassure
    dataset = dataset.shuffle(IMAGENUMBER, reshuffle_each_iteration=True)

    # split Dataset 80/10/10 TODO ugly
    # val_size = int(IMAGENUMBER * 0.1)
    # test_size = int(IMAGENUMBER * 0.1)
    # test_ds = dataset.skip(val_size).take(test_size)
    # train_ds = dataset.skip(val_size + test_size)
    # val_ds = dataset.take(val_size)

    train_frac = .8
    len_train = int(IMAGENUMBER * train_frac)
    # test_frac = (1 - train_frac) / 2
    # val_frac = test_frac
    # train_ds = dataset.range(0, int(IMAGENUMBER * train_frac))
    # val_ds = dataset.range(int(IMAGENUMBER * train_frac), int(IMAGENUMBER * (train_frac + val_frac)))
    # test_ds = dataset.range(int(IMAGENUMBER * (train_frac + val_frac)), IMAGENUMBER)

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(ds=dataset, ds_size=len(all_image_paths), train_split=train_frac)

    print("fullds", dataset.cardinality())
    print("train_ds", train_ds.cardinality())
    print("val_ds", val_ds.cardinality())
    print("test_ds", test_ds.cardinality())

    dataset = train_ds

    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = dataset.shuffle(buffer_size=IMAGENUMBER)
    ds = ds.repeat()
    ds = ds.batch(BATCHSIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    ds = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=IMAGENUMBER))
    ds = ds.batch(BATCHSIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    # # for every datapair [filename, label]:
    # def processImage(datapair):

    #     # argmax encode label: 
    #     label = datapair[1] == classlist
    #     label = tf.argmax(label) # argmax encode label

    #     # prescaled images are used
    #     if not FULLSIZE: 
    #         path = "resized/" + datapair[0]

    #         img = tf.io.read_file(path) # Load the raw data from the file as a string
    #         img = tf.io.decode_image(img, channels=3, expand_animations = False)
    #         img = tf.cast(img, tf.float32)
            
    #         return img, label
        
    #     # images are rescaled during runtime
    #     else:
    #         path = "originalsize/" + datapair[0]

    #         img = tf.io.read_file(path) # Load the raw data from the file as a string
    #         img = tf.io.decode_image(img, channels=3, expand_animations = False)
    #         img = tf.image.resize(img, [IMGSIZE, IMGSIZE])

    #         return img, label

    # # data processing function is mapped to each [path, label combo]
    # train_ds = train_ds.map(processImage, num_parallel_calls=AUTOTUNE)
    # val_ds = val_ds.map(processImage, num_parallel_calls=AUTOTUNE)
    # test_ds = test_ds.map(processImage, num_parallel_calls=AUTOTUNE)

    # # configure for performance # TODO look into this and optimise further
    # def configure_for_performance(ds):
    #     ds = ds.cache()
    #     # ds = ds.shuffle(buffer_size=1000) TODO why shuffle again?

    #     # batch and fetch the next batch after previous one is finished
    #     ds = ds.batch(BATCHSIZE, drop_remainder = True)
    #     ds = ds.prefetch(buffer_size=AUTOTUNE)
    #     return ds

    # # performace function is mapped to each ds
    # train_ds = configure_for_performance(train_ds)
    # val_ds = configure_for_performance(val_ds)
    # test_ds = configure_for_performance(test_ds)

    # define custom model
    '''
    model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=CLASSNUMBER)
    )
    '''
    # define model as list of keras layers
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu", input_shape = (IMGSIZE, IMGSIZE, 3)),
    #     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    #     tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    #     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(100, activation="relu"),
    #     tf.keras.layers.Dense(10, activation="relu")
    #     ])
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
        tf.keras.layers.Dense(10, activation='softmax'),
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
    # model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=EPOCHS,
    #     # callbacks=[cp_callback, WandbCallback()]
    #     callbacks=[cp_callback]
    # )
    model.fit(
        ds,
        epochs=EPOCHS,
        steps_per_epoch=tf.math.ceil(len_train/BATCHSIZE).numpy(),
        callbacks=[cp_callback]
    )
    # after sucessfull run save model
    model.save(MODELPATH)

    # used by confusionmatrix.py so i dont have to implement dynamically getting labels and paths again...
    return [model, classlist, test_ds]

if __name__ == "__main__":
    # input("to compute confusionmatrix start the other script first, continue regardless? press any key ")
    main()
