# if a painting has multiple styles associated with it, we take the most popular one and discard 
import glob
imglist = glob.glob("wikiart-master\saved\images\*\*\*.jpg")
print(len(imglist))
from PIL import Image
import sqlite3

conn = sqlite3.connect("WikiartDataset.db")
c = conn.cursor()

# get list of all unique labels
c.execute("SELECT DISTINCT style FROM artworks")
res = c.fetchall()

orderlist = [] # [genre, number of images]
for genre in res:
    c.execute("SELECT COUNT(style) FROM artworks WHERE style LIKE '%" + genre + "%'")
    res = c.fetchone() 
    orderlist.append([genre, res[0]])

orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)
print(orderlist)

# obtain all classnames as images can contain multiple
classlist = []
for i in res:
    for j in list(i):
        try: # error with "NONE"
            jlist = j.split(",")
        except:
            pass
        for k in jlist:
            if k not in classlist:
                classlist.append(k)

orderlist = [] # [genre, number of images]
for genre in classlist:
    c.execute("SELECT COUNT(style) FROM artworks WHERE style LIKE '%" + genre + "%'")
    res = c.fetchone() 
    orderlist.append([genre, res[0]])

orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)
print(orderlist)


# sort result list
orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

for genre, number in orderlist:
    print(genre)
    c.execute("UPDATE artworks SET style = '" + genre + "' WHERE style LIKE '%" + genre + "%'")

conn.commit()

c.execute("SELECT DISTINCT style FROM artworks")
res = c.fetchall()
print(res)