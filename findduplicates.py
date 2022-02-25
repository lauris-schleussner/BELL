from PIL import Image
import imagehash
from bellutils.getfromdb import getfromdb
from tqdm import tqdm
import hashlib
import sqlite3

global md5
global PUF_GROESSE
md5 = hashlib.md5()
PUF_GROESSE = 1000000     # Puffergröße wird festgelegt

def hashfunc(pfad):

    datei = open("resized/" + pfad , "rb") # Datei wird geöffnet

    # Datei wird stückchenweise gelesen und in einen Hash convertiert
    # https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
    while True:
        daten = datei.read(PUF_GROESSE)
        if not daten:
            break
        md5.update(daten)

    return md5.hexdigest() # hexadezimalzahl als String wird zurückgegeben


conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("SELECT filename FROM artworks WHERE used = True AND corrupt = False")
pathlist = c.fetchall()

hashlist = []
duplicates = []
for p in tqdm(pathlist):
    hash = hashfunc(p[0])
    # print(hash)
    if hash in hashlist:
        duplicates.append(p)
        print("duplicate found")
    else:
        hashlist.append(hash)

print(duplicates)
print(len(hashlist))





