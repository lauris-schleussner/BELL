# resize images using zero padding to 500x500
import os
from bellutils.getfromdb import getfromdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
from tqdm import tqdm

import tensorflow as tf

outpath = "D:/BELL/dataset_resized/"
paths = getfromdb("WikiartDatasetMultiStyles.db")

for imgpath in tqdm(paths):
    
    name = os.path.basename(imgpath)
    try:
        image = tf.io.read_file(imgpath)
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, 500, 500)
        tf.keras.utils.save_img(outpath + name, image)
    except Exception as e:
        print(e)