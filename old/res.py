# create plot of all the different image sizes

import glob
import imagesize
import numpy as np
import matplotlib.pyplot as plt
from pymage_size import get_image_size
import sqlite3
from tqdm import tqdm



# create connection to database
conn = sqlite3.connect("database.db")
c = conn.cursor()

res = c.execute("SELECT filename, style FROM artworks WHERE used = True ORDER BY RANDOM() LIMIT 7517").fetchall()

st = c.execute("SELECT DISTINCT style FROM artworks WHERE used = True").fetchall()

dc = {}
for style in st:
    dc[style[0]] = []


print(dc)
for path, style in tqdm(res):

    img_format = get_image_size("originalsize/" + path)
    w, h = img_format.get_dimensions()
    dc[style].append([w,h])

l = []
colors = ["red","blue","green","grey", "purple"]
avgsize = []
aspectratios = []
for (style, color) in zip(dc, colors):
    xlist = []
    ylist = []
    yges, xges = 0,0

    for x,y in dc[style]:

        # normalise, align along one axis
        if x <= y:
            xlist.append(x)
            ylist.append(y)
            xges += x
            yges += y
        else: 
            xlist.append(x)
            ylist.append(y)
            xges += x
            yges += y
    avgsize.append([style, yges/2517 , xges/2517])

    plt.scatter(xlist, ylist, color = color, alpha = 0.5, label = style)

print(avgsize)

plt.legend(["red - Expressionism", "blue - Realism", "green - Romanticism", "grey - Impressionism", "purple - Art Nouveau" ])
plt.show()


