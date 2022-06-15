import tensorflow as tf
from bellutils.get_datasets import get_datasets


def preprocess_image(filename, label):

    image = tf.io.read_file("C:/Users/Lauris/Downloads/resized/" + filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255 # normalize to [0,1] range

    return image, label

ds = get_datasets("test")
ds = ds.map(preprocess_image)
ds = ds.batch(32)
ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

modeltypes= ["saved_alexnet", "saved_cnn", "saved_resnet50", "saved_resnet50_finetuned", "saved_vgg16", "saved_vgg_16_finetuned", "saved_xception", "saved_xception_finetuned"]

for modeltype in modeltypes:
    modelpath = "D:/BELL/models/06_13_2022_23_45_33/" + modeltype
    model = tf.keras.models.load_model(modelpath)
    model.evaluate(ds, steps = 10)