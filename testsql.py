import sqlite3

DBNAME = "WikiartDataset.db"
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

c.execute("SELECT DISTINCT style FROM artworks")
res = c.fetchall()


# obtain all classnmaes
classnames = []
for i in res:
    for j in list(i):
        try: # error with "NONE"
            jlist = j.split(",")
        except:
            pass
        for k in jlist:
            if k not in classnames:
                classnames.append(k)


# filter out all classes under threshold, takes eternity
for genre in classnames:
    try:
        c.execute("SELECT COUNT(genre) FROM artworks WHERE style LIKE '%" + genre + "%'")
        res = c.fetchone()
        print(res[0])
        if len(res)>= 1000:
            print(genre, len(res))
            classnames.remove(genre)
    except Exception as e:
        print(e)
        pass

print(len(classnames))

