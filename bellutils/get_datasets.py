import sqlite3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
from tensorflow import data
import numpy
import tensorflow as tf

DBNAME = "database.db"

conn = sqlite3.connect(DBNAME)
c = conn.cursor()

def get_datasets(dataset_type):

    assert dataset_type in ["train", "validation", "test"]

    # select list of filenames and styles for the relevant dataset partition
    labels = c.execute("SELECT style FROM artworks WHERE used = True AND " + dataset_type + " = True LIMIT 100").fetchall()
    filenames = c.execute("SELECT filename FROM artworks WHERE used = True AND " + dataset_type + " = True LIMIT 100").fetchall()

    # convert string label to its index
    all_labels = ["Impressionism", "Realism", "Romanticism", "Expressionism", "Art_Nouveau_(Modern)"]
    
    # create lookup dict for all labels
    label_to_index = dict((name, index) for index, name in enumerate(all_labels))

    # use dict to map each style to its index NOTE: all_filenames is needed because filenames is a list in that strange sql return format [(label,), (label2,)]
    # convert to pair for shuffeling
    pair_list = []
    for filepath, label in zip(filenames, labels):
        pair_list.append([filepath[0], label_to_index[label[0]]])
    
    # shuffle 2d pair list
    numpy.random.shuffle(pair_list)

    # convert back to labels and images
    all_index = []
    all_filenames = []
    for filepath, index in pair_list:
        all_index.append(index)
        all_filenames.append(filepath)

    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices((all_filenames, all_index))#

    return dataset

if __name__ == "__main__":
    get_datasets("test")