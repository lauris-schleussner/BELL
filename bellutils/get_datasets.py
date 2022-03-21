import sqlite3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
from tensorflow import data


def get_datasets(dataset_type, DBNAME = "database.db"):

    print("fetching", dataset_type, "from", DBNAME)

    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

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

    print("got", len(all_filenames))

    # create dataset
    dataset = data.Dataset.from_tensor_slices((all_filenames, all_index))

    return dataset

if __name__ == "__main__":
    get_datasets("test")