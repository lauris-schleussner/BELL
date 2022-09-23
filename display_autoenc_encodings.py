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
from tqdm import tqdm
from functools import cache

# db
conn = sqlite3.connect("database.db")

# use sqlite memory db to improve speed
source = sqlite3.connect("database.db")
conn = sqlite3.connect(':memory:')
source.backup(conn)

c = conn.cursor()

# load encodings
encodings = pickle.load(open("image_embeddings_16_ep.pickle", "rb" ))

# extract list of all files and encodings
all_files = encodings["indices"]
all_encodings = encodings["features"]


all_encodings_pre = all_encodings[0:5000]
all_files = all_files[0:5000]


relevant_styles = ["Impressionism","Realism",
"Romanticism",
"Expressionism",
"Art_Nouveau_(Modern)",
"Post_Impressionism",
"Baroque",
"Symbolism"]

# get all data and check if style is relevant or none, if not add to list, else write "None" to list
@cache
def getallstyles():
    all_styles = []
    for file in tqdm(all_files):
        style = c.execute("SELECT style from artworks WHERE filename ='" + file + "'").fetchone()
        if style is not None and style[0] in relevant_styles:

            all_styles.append(style[0])

        else:
            pass
            all_styles.append("None")

    return all_styles

all_styles_pre = getallstyles()


# filter out none
all_encodings = []
all_styles = []
for emb, style in zip(all_encodings_pre, all_styles_pre):
    if style != "None":
        all_encodings.append(emb)
        all_styles.append(style)



if True:

    # reduce encoding dimensions to 30
    #pca = PCA(n_components=15)
    #reduced_encodings = pca.fit_transform(all_encodings)

    # init tsne
    tsne = TSNE(random_state = 1, n_components = 2, perplexity = 100, n_iter = 2000, verbose = 3, early_exaggeration=50, init = "pca")

    # get tsne embeds
    embs = tsne.fit_transform(all_encodings)

    # Add to dataframe for convenience
    df = pd.DataFrame
    all_x = embs[:, 0]
    all_y = embs[:, 1]
    
    
else:

    all_encodings = StandardScaler().fit_transform(all_encodings)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2)
    embs = reducer.fit_transform(all_encodings)
    all_x = embs[:, 0]
    all_y = embs[:, 1]

# dump 2d points
data = {'points': [all_x, all_y], 'style': all_styles}
pickle.dump(data, open('img_embed_autoenc_afterumap.pickle', 'wb'))

# get unique styles
unique_styles = []
for style in tqdm(all_styles):
    print(style)
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


scatter = plt.scatter(all_x, all_y, c = colornumber, alpha = 0.7)

li = []
for i in range(len(all_x)):
    li.append(i)

# tooltip logic

for x, y, txt, i in zip(all_x, all_y, all_files, li):
    # not annotate everything
    if i % 100 == 0:
        plt.annotate(txt, (x,y))


plt.xticks([])
plt.yticks([])
plt.legend(handles=scatter.legend_elements()[0], labels=unique_styles, fontsize = 13)
plt.show()