# für einen gegebenen Pfad wird ein average-hash zurückgegeben 

from PIL import Image
import imagehash    

def main(pfad):
    
    bild = Image.open(pfad)
    ahash = imagehash.average_hash(bild, hash_size=10) # Größe des hashes wird festgelegt (10 Zeichen lang)

    return str(ahash)