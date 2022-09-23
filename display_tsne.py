import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import sqlite3
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import StandardScaler
import vaex

# db
conn = sqlite3.connect("database.db")
c = conn.cursor()
# load encodings
encodings = pickle.load(open("image_embeddings_32_ep.pickle", "rb" ))

# extract list of all files and encodings
all_files_and_styles = encodings["indices"]
all_encodings = encodings["features"]


all_encodings = all_encodings[0:3000]
all_files_and_styles = all_files_and_styles[0:3000]

print(all_encodings.shape)

if False:

    # reduce encoding dimensions to 30
    pca = PCA(n_components=15)
    reduced_encodings = pca.fit_transform(all_encodings)

    # init tsne
    tsne = TSNE(random_state = 1, n_components = 2, perplexity = 40, n_iter = 1000, verbose = 3, early_exaggeration=50, init = "pca")

    # get tsne embeds
    embs = tsne.fit_transform(all_encodings)

    # Add to dataframe for convenience
    df = pd.DataFrame
    all_x = embs[:, 0]
    all_y = embs[:, 1]
    
    
else:

    all_encodings = StandardScaler().fit_transform(all_encodings)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2)
    embs = reducer.fit_transform(all_encodings)
    all_x = embs[:, 0]
    all_y = embs[:, 1]

all_styles = []
all_files = []
for files, styles in all_files_and_styles:
    all_styles.append(styles)
    all_files.append(files)

# dump 2d points
#data = {'points': [all_x, all_y], 'style': all_styles}
#pickle.dump(data, open('img_embed_xception_after_umap.pickle', 'wb'))

# get unique styles
unique_styles = []
for style in tqdm(all_styles):
    if style not in unique_styles:
        unique_styles.append(style)   

colordict = {}

# create nlist of numbers
a = []
for i in range(len(unique_styles)):
    a.append(i)

# create color dict
for  i, style in zip(a, unique_styles):
    colordict[style] =  i

# convert style names to numbers
colornumber = []
for style in all_styles:
    colornumber.append(colordict[style])


scatter = plt.scatter(all_x, all_y, c = colordict, alpha = 0.7)

li = []
for i in range(len(all_x)):
    li.append(i)

# tooltip logic
"""
for x, y, txt, i in zip(all_x, all_y, all_files, li):
    # not annotate everything
    if i % 50 == 0:
        plt.annotate(txt, (x,y))
"""

plt.xticks([])
plt.yticks([])
plt.legend(handles=scatter.legend_elements()[0], labels=unique_styles, fontsize = 13)
plt.show()