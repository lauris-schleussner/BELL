import sqlite3

DBNAME = "wikiartDataset.db"
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

c.execute("SELECT genre FROM artworks WHERE genre IS NOT NULL")
res = c.fetchall()

print(len(res))

# Genre spalte kann mehrere enthalten
# werden gesplittet
unique = []
for i in res:
    for j in list(i):
        jlist = j.split(",") # splitten in einzelne genres
        for k in jlist:
            if k not in unique:
                unique.append(k)

print(unique)

for genre in unique:
    try:
        c.execute("SELECT genre FROM artworks WHERE genre LIKE '%" + genre + "%'")
        res = c.fetchall()
        if len(res)>= 200:
            print(genre, len(res))
    except Exception as e:
        pass

