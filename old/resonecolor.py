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

res = c.execute("SELECT filename FROM artworks WHERE corrupt= False").fetchall()

filenames = []
for i in res:
    filenames.append(i[0])


wlist = []
hlist = []
ratio = []
for path  in tqdm(filenames):

    img_format = get_image_size("originalsize/" + path)
    w, h = img_format.get_dimensions()
    wlist.append(w)
    hlist.append(h)

    # 
    ratio.append(max([w,h])/min([w,h]))




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


