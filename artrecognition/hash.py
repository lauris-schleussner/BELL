# für einen gegebenen Pfad wird ein md5 Hash zurückgegeben 

import hashlib

def main(pfad):

    md5 = hashlib.md5()     # Hash Algorithmus wird festgelegt
    PUF_GROESSE = 10000     # Puffergröße wird festgelegt

    datei = open(pfad, "rb") # Datei wird geöffnet

    # Datei wird stückchenweise gelesen und in einen Hash convertiert
    # https://stackoverflow.com/questions/1131220/get-md5-hash-of-big-files-in-python
    while True:
        daten = datei.read(PUF_GROESSE)
        if not daten:
            break
        md5.update(daten)

    return md5.hexdigest() # hexadezimalzahl als String wird zurückgegeben