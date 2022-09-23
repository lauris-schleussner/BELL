# read autoencoder embeddings and find most similar images 

import pickle
from scipy.spatial import distance
from PIL import Image
from tqdm import tqdm


# filename to which simmilar images will be found
filename_to_compare = "185131.jpg"

# open encodings produced from similar_autoencoder.py
encodings = pickle.load(open("E:\BELL\img_embed_autoenc_afterumap.pickle", "rb" ))

# extract list of all files and encodings
all_files = encodings["indices"]
all_encodings = encodings["features"]

# get dict index of the file that is compared against
to_compare_index = all_files.index(filename_to_compare)

# get the encoding of the file that is used to compare against
to_compare_encodings = all_encodings[to_compare_index]

# check every other encoding against the main one
filename_dist_dict = {}
for encoding, index in tqdm(zip(all_encodings, range(0,len(all_encodings)))):

    # get distance to base image
    # dist = np.linalg.norm(to_compare_encodings[index], encoding)
    dist = distance.euclidean(encoding, to_compare_encodings)

    filename_dist_dict[index] = dist


# sort by distance
sorted_dict = dict(sorted(filename_dist_dict.items(), key=lambda item: item[1]))

# get first {topnumber} matches
topnumber = 9
firstXpairs = {k: sorted_dict[k] for k in list(sorted_dict)[:topnumber]}

# list all indices
matching_indices = []
for i in firstXpairs:
    matching_indices.append(i)

# list matching filenames
matchingfilenames = []
for i in matching_indices:
    matchingfilenames.append("resized/" + all_files[i])
    print(all_files[i])

# display matches. First match is always the original
for i in matchingfilenames:
    image = Image.open(i)
    image.show()


