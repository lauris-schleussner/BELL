import glob
import imagesize
import numpy as np
import matplotlib.pyplot as plt
from pymage_size import get_image_size
import sqlite3
from tqdm import tqdm
import matplotlib.lines as mlines


# create connection to database
conn = sqlite3.connect("database.db")
c = conn.cursor()

res = c.execute("SELECT filename, style FROM artworks WHERE corrupt= False and used = True").fetchall()

sizedict = {"Impressionism": [], "Realism": [], "Romanticism " :  [], "Expressionism" : [], "Art_Nouveau_(Modern)" : []}

data = []
for file, style in res:
    data.append([file, style])


wlist = []
hlist = []
ratio = []
for path, style in tqdm(data):

    img_format = get_image_size("originalsize/" + path)
    w, h = img_format.get_dimensions() 
    sizedict[style].append(max([w,h])/min([w,h]))

# init plot
fig, axs = plt.subplots(1,5)
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()


for idx, ax in enumerate(axs):
    stylelist = list(sizedict.keys())
    style = stylelist[idx]

    ax.hist(sizedict[style], density = True,  bins = 100, range = (1,2))
    ax.set_title(style)
    



plt.show()
quit()


print(sizedict)

plt.hist(ratio, density = True,  bins = 100, range = (1,2))
plt.title("Seitenverhältnis")
plt.yticks([]) 


"""
plt.scatter(wlist, hlist)

plt.xlim(0, 5000)
plt.ylim(0, 5000)

plt.xlim(0, 5000)
plt.ylim(0, 5000)
plt.gca().set_aspect('equal', adjustable='box')




plt.title("Bildgröße")
plt.xlabel("Breite in px")
plt.ylabel("Höhe in px")
"""
plt.show()
