# TODO implement SSIM, PSNR

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
import sqlite3

# network parameter settings
BATCHSIZE = 32
LEARNINGRATE = 0.001 # default Adam learning rate
IMGSIZE = 244 # images are rescaled to a square, size in px

# paths
MODELPATH = "models/"
DBNAME = "database.db"
CPPATH = "models/checkpoint.ckpt"
CPDIR = os.path.dirname(CPPATH)


# database
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

# Tensorflow Stuff 
AUTOTUNE = tf.data.AUTOTUNE

# return dataset for requested partition (train/val/test)
def get_datasets(dataset_type):

    assert dataset_type in ["train", "validation", "test"]

    # select list of filenames and styles for the relevant dataset partition
    labels = c.execute("SELECT style FROM artworks WHERE used = True AND " + dataset_type + " = True").fetchall()
    filenames = c.execute("SELECT filename FROM artworks WHERE used = True AND " + dataset_type + " = True").fetchall()

    # convert string label to its index
    all_labels = ["Impressionism", "Realism", "Romanticism", "Expressionism", "Art_Nouveau_(Modern)"]
    
    # create lookup dict for all labels
    label_to_index = dict((name, index) for index, name in enumerate(all_labels))

    # use dict to map each style to its index NOTE: all_filenames is needed because filenames is a list in that strange sql return format [(label,), (label2,)]
    all_index = []
    all_filenames = []
    for filepath, label in zip(filenames, labels):
        all_index.append(label_to_index[label[0]])
        all_filenames.append(filepath[0])

    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_filenames, all_index))

    return dataset


# open and read image. This function is mapped to every dataset row
def preprocess_image(filename, label):

    image = tf.io.read_file("resized/" + filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255 # normalize to [0,1] range

    return image, label


def main(modelname, EPOCHS):

    TRAININGMODELPATH = "./untrained/" + modelname

    # get dataset
    train_ds = get_datasets("train")
    val_ds = get_datasets("validation")
    test_ds = get_datasets("test")

    # load and preprocess images
    train_ds = train_ds.map(preprocess_image)
    val_ds = val_ds.map(preprocess_image)
    test_ds = test_ds.map(preprocess_image)


    # shuffle datasets
    train_ds = train_ds.shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)
    val_ds = val_ds.shuffle(val_ds.cardinality(), reshuffle_each_iteration=True)

    # info
    print("train_ds", train_ds.cardinality())
    print("val_ds", val_ds.cardinality())
    print("test_ds", test_ds.cardinality())

    train_ds = train_ds.batch(BATCHSIZE)
    val_ds = val_ds.batch(BATCHSIZE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # load model from file created by "createmodels.py"
    # model = tf.keras.models.load_model(MODELPATH)

    
    # alexnet
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

    model.summary()

    # "Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses."
    # gradient descent to improve training
    optimizer = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

    # configure model for training
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # callbacks that are triggered during training, create checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CPPATH, save_weights_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[cp_callback, es_callback]
    )
    # after sucessfull run save model
    model.save(MODELPATH)

    print("succesfully trained and saved the model ", modelname)

    return [model, history, test_ds]

if __name__ == "__main__":
    main()
