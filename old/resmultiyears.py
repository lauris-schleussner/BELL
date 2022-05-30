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

res = c.execute("SELECT filename, path FROM artworks WHERE corrupt = False").fetchall()

bins = np.array([1750, 1770, 1790, 1810,1830,1850,1870,1890,1910,1930, 1950])

sizedict = {}

# creaste dict keys for each year
for yearbin in bins:
    sizedict[yearbin] = []

data = []
for file, year in res:
    if year is None or int(year) <= 1725 or int(year)>= 1975:
        continue

    data.append([file, year])

for path, year in tqdm(data):

    img_format = get_image_size("originalsize/" + path)
    w, h = img_format.get_dimensions() 

    # find which epoch from the bins should be taken
    dict_key = min(bins, key=lambda x:abs(x-int(year)))

    # calculate size and add to list
    sizedict[dict_key].append(max([w,h])/min([w,h]))

# init plot
fig, axs = plt.subplots(1,len(bins))
fig.subplots_adjust(hspace = .5, wspace=.2)
axs = axs.ravel()


for idx, ax in enumerate(axs):
    yearlist = list(sizedict.keys())
    year = yearlist[idx]

    ax.hist(sizedict[year], density = True,  bins = 50, range = (1,2))
    ax.set_title(str(year) + "  n=" + str(len(sizedict[year])))
    ax.set_yticks([]) 
    



plt.show()
quit()