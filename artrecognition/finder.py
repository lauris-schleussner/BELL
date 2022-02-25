# für einen gegebenen Pfad gibt main() eine liste aller jeweiligen ergebnisse [hash, ahash, sift] zurück
# Das Ergebnis jeder der drei Suchfunktionen ist entweder die Bild_ID oder False 

import hash
import ahash
import orbfex
import cv2
import sqlite3
import copyreg
import pickle

# ORB Objekt wird global deklariert
orb_obj = cv2.ORB_create()

# Datenbankverbindung wird hergestellt
con = sqlite3.connect('db.db')
cur = con.cursor()

# Funktion um Keypoints zu deserialisieren
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

# gibt die Ergebnisse zurück
def main(pfad):

    hashreturn = hashsearch(pfad)
    if hashreturn:
        return hashreturn

    ahashreturn = ahashsearch(pfad)
    if ahashreturn:
        print("ahash")
        return ahashreturn

    orbreturn = orbsearch(pfad)
    if orbreturn:
        return orbreturn
    
    return False

# Suche mithilfe des md5 Hash einer Datei
def hashsearch(pfad):
    
    # Hash des Originalen (Input) Bildes wird generiert
    orig_hash = hash.main(pfad) 

    # Datenbank wird nach exakten Hash durchsucht
    cur.execute("SELECT id FROM kunstwerke WHERE hash = '%s'" % orig_hash)
    result = cur.fetchone() # gibt ein Tuple zurück

    # wenn erfolgreich wird ID des Bildes zurückgegeben
    if result != None:
        return result[0]
    else:
        return False   

# durchsuche die Datenbank mithilfe eines A-Hash
def ahashsearch(pfad):

    orig_ahash = ahash.main(pfad)

    # Datenbank wird nach exakten A-Hash durchsucht
    cur.execute("SELECT id FROM kunstwerke WHERE ahash = '%s'" % orig_ahash)
    result = cur.fetchone() # gibt ein Tuple zurück
        
    # wenn erfolgreich wird ID des Bildes zurückgegeben
    if result != None:
        return result[0]
    else:
        return False   

# Durchsuche Datenbank von ORB Deskriptoren
def orbsearch(pfad):

    orig_orb = orbfex.main(pfad, orb_obj, rpickle = False)

    # FLANN Parameter werden festgelegt
    # für ORB deskriptoren wird FLANN_Index_LSH genutzt
    # https://stackoverflow.com/questions/29563429/opencv-python-fast-way-to-match-a-picture-with-a-database
    FLANN_INDEX_LSH = 6
    indexparameter= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, 
                   key_size = 12,     
                   multi_probe_level = 1)
    suchparameter = dict(checks=50)

    # FLANN Objekt wird initiiert 
    flann = cv2.FlannBasedMatcher(indexparameter, suchparameter)


    # alle ORB Deskriptoren werden aus der Datenbank geladen
    for vgl_orb in cur.execute("SELECT orb FROM kunstwerke"):

        # Keypoints werden deserialisiert
        vgl_orb = pickle.loads(vgl_orb[0]) 

        # Deskriptoren jedes Keypoints werden zum FLANN Index hinzugefügt
        flann.add([vgl_orb[1]])

    # FLANN wird trainiert
    flann.train()  
    
    # Deskriptoren der Datenbank werden mit Original Deskriptoren verglichen
    # ein DMatch Objekt wird generiert
    matches = flann.knnMatch(orig_orb[1], k=2)

    # Verhältnis Test nach LOWE
    # Gute Übereinstimmungen werden in Liste gespeichert
    # Jedes DMatch Objekt besitzt die "distance" Eigenschaft als Wert, wie stark zwei Deskriptoren übereinstimmen 
    # https://www.docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    gute_matches = []
    for m,n in matches:
        if m.distance < 0.9 * n.distance:
            gute_matches.append(m)

    # werden keine guten Übereinstimmungen gefunden wird False zurückgegeben
    if len(gute_matches) <= 3:
        return False

    # Dictionary wird erstellt: [bildzahl : distanz summe]
    matchdict = {}
    # Zählt die Anzahl der übereinstimmenden Deskriptoren pro Bild
    iddict = {}

    # Deskriptoren werden ihrem Originalbild wieder zugeordnet
    # Die Anzahl der passenden Deskriptoren pro Bild wird gezählt
    for deskriptor in gute_matches:
        
        # Wenn der Dictionary Eintrag noch nicht existiert wird er erstellt
        try:
            iddict[deskriptor.imgIdx] += 1
            matchdict[deskriptor.imgIdx] +=deskriptor.distance
        except:
            iddict[deskriptor.imgIdx] = 1
            matchdict[deskriptor.imgIdx] = deskriptor.distance

    # Distanz pro Bild wird berechnet
    for bildid in matchdict:

        # Besitzt ein Bild zu wenig Deskriptoren wird eine sehr hohe Distanz berechnet
        # damit es so imm nächsten Schritt aussortiert wird
        if iddict[bildid] <= 2:
            matchdict[bildid] = 99999999
        else:
            # gesamtdistanz wird durch anzahl der passenden Deskriptoren geteilt
            matchdict[bildid] = matchdict[bildid] / iddict[bildid]

    # Das Dictionary wird nach der kleinsten Distanz sortiert
    matchdict = sorted(matchdict.items(), key=lambda x: x[1])

    # Ist die Distanz des Besten Ergebnisses zu hoch, wird False zurückgegeben
    if matchdict[0][1] >= 10000:
        return False
    else:
        return matchdict[0][0] + 1 # Datenbank Index unterscheidet sich um 1 vom DMatch Index