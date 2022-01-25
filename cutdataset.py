# adjust dataset so that every class has the same number of images. Images are picked randomly from bigger classes'''
# is can also be set which classes should be used (have the "used" flag set)'''
# missing images are also removed'''

from PIL import Image
import os
from tqdm import tqdm
import sqlite3
import numpy as np

# styles that will be used. 
relevantstyles = ["Impressionism", "Realism", "Romanticism", "Expressionism","Art_Nouveau_(Modern)"]

# connect to db
conn = sqlite3.connect("database.db")
c = conn.cursor()

# set used flags for styles again, this time correctly
c.execute("SELECT id FROM artworks")
ids = c.fetchall()
for id in tqdm(ids):

    # check every style
    c.execute("SELECT style FROM artworks where id = '" + str(id[0]) + "'")
    style = c.fetchall()[0][0]
    if style in relevantstyles:
        c.execute("UPDATE artworks SET used = True WHERE id = '" + str(id[0]) + "'")
    else:
        c.execute("UPDATE artworks SET used = False WHERE id = '" + str(id[0]) + "'")

# adjust class size:
# get list of all unique styles
c.execute("SELECT DISTINCT style FROM artworks WHERE used = True")
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

# sort by number of images each genre has
orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

# determine amount of images in "smallest" style
minclasssize = orderlist[-1][1]
print("images per class:", minclasssize)

# fetch ids of images that were selected
# The shuffeling is done before actually selecting, so these images *should* be random TODO verify
res = []
for classname in classlist:
    querry = "SELECT id FROM artworks WHERE style = '" + classname + "' ORDER BY RANDOM() LIMIT " + str(minclasssize)
    c.execute(querry)
    res += c.fetchall()

# clean up list of all images that were selected
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

# ids that should be set to "unused" (ids that where not selected because a class is to big)
# yields the elements in `fullidlist` that are NOT in `selectidlist`
deleteids = np.setdiff1d(fullidlist, selectidlist)

print("full", len(fullidlist))
print("selected", len(selectidlist))
print("set unused", len(deleteids))

# update db so each class has the same size
for id in tqdm(deleteids):
    c.execute("UPDATE artworks SET used = false WHERE id = '" + str(id) + "'")
    res = c.fetchall()

# save changes
conn.commit()

# check again, if all images do actually exist. Some go missing in the cleaning process
c.execute("SELECT filename style from artworks WHERE used = True")
res = c.fetchall()
list = []
for i in res:
    list.append(i[0])

# and... delete
for p in tqdm(list):
    if not os.path.isfile("resized/" + str(p)) or not os.path.isfile("resized/" + str(p)):
        c.execute("UPDATE artworks SET used = false WHERE filename = '" + str(p) + "'")

# save again and close connection
conn.commit()
conn.close()