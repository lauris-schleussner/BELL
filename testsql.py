import sqlite3
import numpy


DBNAME = "database.db"
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

c.execute("SELECT style FROM artworks WHERE validation = True")
res = c.fetchall()
for i in res:
    pass
    print(i)


# calculate mixedness


import random
# random.shuffle(res)



# calculate the sum of the lengths of all streaks against the length of the dataset
# around 0.9 is good
# under 0.7 is bad
MINLENGTH = 2
counter = 0
li = ["A", "B", "C", "D"]
for i in res:
    a = i[0]
    if li[-1] == a:
        li.append(a)
    else:
        max = len(li)
        if max <= MINLENGTH:
            counter += max
        li = [a]

print(counter/len(res))