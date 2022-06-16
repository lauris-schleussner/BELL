import pickle
import numpy
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.manifold import TSNE

encodings = pickle.load(open("img_embed_xception.pickle", "rb" ))

# extract list of all files and encodings
all_files = encodings["indices"]
all_encodings = encodings["features"]

tsne = TSNE(random_state=1)

embs = tsne.fit_transform(all_encodings)
# Add to dataframe for convenience
df = pd.DataFrame

all_x = embs[:, 0]
all_y = embs[:, 1]

import sqlite3
conn = sqlite3.connect("database.db")
c = conn.cursor()

def getstyle(path):
    filename = path[-10:]
    print(filename)

    style = c.execute("SELECT style FROM artworks WHERE filename = '" + str(filename) + "'").fetchall()
    print(style)
    return style

all_styles = []
unique_styles = []
for i in tqdm(all_files):
    style = getstyle(i)
    if style not in unique_styles:
        unique_styles.append(style)   

    all_styles.append(style)

print(unique_styles)

plt.scatter(all_x, all_y, c=all_styles)
plt.show()

