import sqlite3
import numpy


DBNAME = "database.db"
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

a = c.execute("SELECT filename FROM artworks LIMIT 1").fetchall()
print(a)
