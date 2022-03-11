import pickle
import numpy as np
import pandas as pd
import sqlite3
from scipy.spatial import distance
import matplotlib.pyplot as plt
from PIL import Image

# connect to db
conn = sqlite3.connect("database.db")
c = conn.cursor()

# filename to which simmilar images will be found
filename_to_compare = "190420.jpg"

# open encodings produced from similar_autoencoder.py
encodings = pickle.load(open("autoenc\image_embeddings.pickle", "rb" ))

# extract list of all files and encodings
all_files = encodings["indices"]
all_encodings = encodings["features"]

# get index of the file that is compared against
to_compare_index = all_files.index(filename_to_compare)

# get the encoding of the file that is used to compare against
to_compare_encodings = all_encodings[to_compare_index]

# check every other encoding against the main one
filename_dist_dict = {}
for encoding, index in zip(all_encodings, range(0,len(all_encodings))):


    # get distance to base image
    # dist = np.linalg.norm(to_compare_encodings[index], encoding)
    dist = distance.euclidean(encoding, to_compare_encodings)

    filename_dist_dict[index] = dist


# sort by distance
sorted_dict = dict(sorted(filename_dist_dict.items(), key=lambda item: item[1]))

# get first 10 matches
first10pairs = {k: sorted_dict[k] for k in list(sorted_dict)[:10]}

# list all indices
matching_indices = []
for i in first10pairs:
    matching_indices.append(i)

# list matching filenames
matchingfilenames = []
for i in matching_indices:
    matchingfilenames.append("resized/" + all_files[i])
    print(all_files[i])

# display
for i in matchingfilenames:
    image = Image.open(i)
    image.show()
