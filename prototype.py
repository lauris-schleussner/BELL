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
    BATCHSIZE = 10
    EPOCHS = 1
    LEARNINGRATE = 0.001 # default Adam learning rate
    IMGPERCLASS = 100 # 1600 # how many images there should be per class. # TODO determine this
    CLASSNUMBER = 4 # should be around 10-20 to make sense
    IMAGENUMBER = IMGPERCLASS * CLASSNUMBER # 

    # paths
    MODELPATH = "D:/BELL/models"
    checkpoint_path = "models/checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    DBNAME = "WikiartDataset.db"

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
            print(i[0])

    # filter out None Types that appear
    classlist = filter(None.__ne__, classlist)

    # filter out all classes under threshold
    orderlist = [] # ["style", "number of images"]
    for style in classlist:
        c.execute("SELECT COUNT(style) FROM artworks WHERE style = '" + style + "'")
        amount = c.fetchone() 
        orderlist.append([style, amount[0]])

    # sort result list by number of images each genre has
    orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

    # take specified number of styles that have the most images
    orderlist = orderlist[0:CLASSNUMBER]

    # determine amount of images in "smallest" style
    # = smallest possible class size as each class must have the same size
    # if smallest possible class size is surpassed, IMGPERCLASS is set to the max possible amount
    maxPossiblePerClass = orderlist[-1][1]
    if IMGPERCLASS >= maxPossiblePerClass:
        IMGPERCLASS = maxPossiblePerClass
        print("Too many images per class specified, taking max possible amount")
    print("images per class:", IMGPERCLASS)

    # get list of all classes without their size
    classlist = []
    for style in orderlist:
        classlist.append(style[0])

    print("classnumber: ", len(classlist))
    print("classes:", classlist)

    # fetch querry
    # randomly selects IMGPERCLASS-many mages from each class
    # The shuffeling is done before actually selecting, so these images *should* be random im IMGPERCLASS is smaller than the actual amount of images TODO verify
    # the path is added to a long list
    res = []
    for classname in classlist:
        querry = "SELECT path, style FROM artworks WHERE style = '" + classname + "' ORDER BY RANDOM() LIMIT " + str(IMGPERCLASS)
        c.execute(querry)
        res += c.fetchall()

    # create dataset from list of all paths
    dataset = tf.data.Dataset.from_tensor_slices(res)

    # shuffle dataset TODO might be faster to numpy-shuffle the pathlist first
    # images in each class are shuffled from SQL querry but not the classes itself
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

    # for every datapair [path, label]:
    def processImage(datapair):
        # TODO determine
        height = 150
        width = 150

        # argmax encode label: 
        label = datapair[1] == classlist
        label = tf.argmax(label) # argmax encode label

        # decode and resize image
        img = tf.io.read_file(datapair[0]) # Load the raw data from the file as a string
        img = tf.io.decode_image(img, channels=3, expand_animations = False)
        img = tf.image.resize(img, [height, width])

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
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(CLASSNUMBER)
        ])

    # "Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses."
    # gradient descent to improve training
    opt = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

    # configure model for training
    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), # TODO shouldnt the optimizer take care of this?
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
    return [model, orderlist, test_ds]

if __name__ == "__main__":
    input("to compute confusionmatrix start the other script first, continue regardless? press any key")
    main()
