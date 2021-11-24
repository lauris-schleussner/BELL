import sqlite3
import os

conn = sqlite3.connect("WikiartDatasetMultiStyles.db")
c = conn.cursor()

c.execute("SELECT id, path FROM artworks WHERE style = 'Impressionism' OR style = 'Realism' OR style = 'Romanticism' OR style = 'Expressionism' OR style = 'Art_Nouveau_(Modern)' OR style = 'Post_Impressionism' OR style = 'Baroque'")
res = c.fetchall()
print(len(res))
# clean up
returnlist = []
for i in res:
    returnlist.append([i[0], i[1]])

# change path
for id, path in returnlist:
    name = os.path.basename(path)
    dir = "dataset_resized"
    c.execute("UPDATE artists SET path = " + str(dir) + str(name) + "WHERE id = " + str(id))

sql = "UPDATE artworks SET path = REPLACE(path,) "

c.execute(sql)
# conn.commit()
print(c.rowcount, "record(s) affected")