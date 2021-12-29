'''adjust dataset so that every class has the same number of images. Images are picked randomly from bigger classes'''

from PIL import Image
import os
from tqdm import tqdm
import sqlite3
import numpy as np

DBPATH = "database.db"

# connect
conn = sqlite3.connect(DBPATH)
c = conn.cursor()

# get list of all unique styles returns list of tupels with one datapair each
c.execute("SELECT DISTINCT style FROM artworks")
res = c.fetchall()

# convert list of tupels to clean list of strings
classlist = []
for i in res:
        classlist.append(i[0])


# filter out all classes under threshold
orderlist = [] # ["style", "number of images"]
for style in classlist:
    c.execute("SELECT COUNT(style) FROM artworks WHERE style = '" + style + "'")
    amount = c.fetchone() 
    orderlist.append([style, amount[0]])


# sort result list by number of images each genre has
orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)



# determine amount of images in "smallest" style
# = smallest possible class size as each class must have the same size
# if smallest possible class size is surpassed, IMGPERCLASS is set to the max possible amount
minclasssize = orderlist[-1][1]
print("images per class:", minclasssize)

# fetch ids of images that were selected
# The shuffeling is done before actually selecting, so these images *should* be random if IMGPERCLASS is smaller than the actual amount of images TODO verify
res = []
for classname in classlist:
    querry = "SELECT id FROM artworks WHERE style = '" + classname + "' ORDER BY RANDOM() LIMIT " + str(minclasssize)
    c.execute(querry)
    res += c.fetchall()

# clean res
selectidlist = []
for id in res:
    selectidlist.append(id[0])

# select full list
querry = c.execute("SELECT id FROM artworks")
res = c.fetchall()

# clean full list
fullidlist = []
for id in res:
    fullidlist.append(id[0])


# ids that should be deleted (ids that where not selected because a class is to big)
# yields the elements in `fullidlist` that are NOT in `selectidlist`
deleteids = np.setdiff1d(fullidlist, selectidlist)

print("full", len(fullidlist))
print("selected", len(selectidlist))
print("delete", len(deleteids))

for id in tqdm(deleteids):
    querry = c.execute("DELETE FROM artworks WHERE id = '" + str(id) + "'")
    res = c.fetchall()

conn.commit()
conn.close()

conn = sqlite3.connect(DBPATH)
c = conn.cursor()

querry = c.execute("SELECT id FROM artorks")
res = c.fetchall()
print(len(res))