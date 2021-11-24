# get a list of all imagepaths that are contained in the database

import sqlite3

def getfromdb(DBNAME):
    path = "D:/BELL/" + DBNAME

    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT path FROM artworks")
    res = c.fetchall()
    
    # clean up
    returnlist = []
    for i in res:
        returnlist.append(i[0])

    return returnlist

if __name__ == "__main__":
    DBNAME = "WikiartDatasetMultiStyles.db"
    getfromdb(DBNAME)