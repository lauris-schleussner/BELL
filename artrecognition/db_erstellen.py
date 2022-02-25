# erstellt eine mysqli Datenbank aus den heruntergeladenen Wikiart-Daten

import json
import hash
import ahash
import orbfex
import sqlite3
import cv2
import sqlite3
from tqdm import tqdm

DATENBANKNAME = "duplicates.db"

# verbinde mit Datenbank
conn = sqlite3.connect(DATENBANKNAME)
c = conn.cursor()

# ORB Objekt wird nur einmal global erstellt und an orbfex.py übergeben
orb_obj = cv2.ORB_create()

def erstellen():
    # KünstlerTabelle wird erstellt
    c.execute("""CREATE TABLE IF NOT EXISTS kuenstler (
                id integer PRIMARY KEY,
                name text NOT NULL,
                geburtsdatum text,
                sterbedatum text,
                wikipedia text
                )""")
    # Kunstwerke Tabelle wird erstellt
    c.execute("""CREATE TABLE IF NOT EXISTS kunstwerke (
                id integer PRIMARY KEY,
                bild_zahl integer,
                titel text NOT NULL,
                name text,
                pfad text,
                jahr text,
                link text,
                ort text,
                material text,
                genre text,
                stil text,
                preis real,
                galerie_name text,
                hash text,
                ahash text,
                orb blob
                )""")

def fuellen():
    # artists.json enthält eine Liste aller Künstler 
    with open("wikiart-master/saved/meta/artists.json", encoding="latin-1") as f:
        json_datei = json.load(f)

    # "kuenstler" ist ein Python dictionary 
    for kuenstler in tqdm(json_datei):

            # Die Dictionary Werte werden in Variabeln eingelesen
            kuenstler_url = kuenstler["url"]
            name = kuenstler["artistName"]
            geburtsdatum = kuenstler["birthDayAsString"]
            sterbedatum = kuenstler["deathDayAsString"]
            wikipedia = kuenstler["wikipediaUrl"]

            # Werte werden in die Künstler Tabelle eingefügt
            sql = "INSERT INTO kuenstler(name, geburtsdatum, sterbedatum, wikipedia) VALUES (?, ?, ?, ?)"
            val = [
                (name, geburtsdatum, sterbedatum, wikipedia)
            ]
            c.executemany(sql, val)

            # sollte artists.json einen Künstler enthalten der keine eigene JSON Datei besitzt wird dieser Künstler übersprungen
            try:

                # eigene JSON Datei der Künstler wird geöffnet
                with open("wikiart-master/saved//meta/" + kuenstler_url + ".json", encoding="latin-1") as f:
                    bilddaten_json = json.load(f)

            except Exception as e:
                print(e)
                continue

            # Fortschrittsanzeige
            bildzaehler = 0

            # für jedes Bild des Künstlers
            for bild in bilddaten_json:

                try:

                    # Fortschrittsanzeige
                    bildzaehler += 1
                    
                    # Pfad des jeweiligen Bildes wird zusammengesetzt
                    # Pfad: "wikiart-saved/images/-artistUrl-/-year-/-contentId-"
                    # Jahreszahl "0" wird zu "unknown-year" im Pfad
                    if len(bild["yearAsString"]) == 0:
                        jahr = "unknown-year"
                    else:
                        jahr = bild["yearAsString"]
                    pfad = "wikiart-master/saved/images/" + str(bild["artistUrl"]) + "/" + jahr +"/" + str(bild["contentId"]) + ".jpg"

                    # Informationen werden für jedes Bild gesammelt
                    bild_zahl = bild["contentId"]
                    titel = bild["title"]
                    jahr = bild["yearAsString"]
                    link = bild["image"]
                    ort = bild["location"]
                    material = bild["material"]
                    genre = bild["genre"]
                    stil = bild["style"]
                    preis = bild["lastPrice"]
                    galerie_name = bild["galleryName"]
                    hash_return = hash.main(pfad)
                    ahash_return = ahash.main(pfad)
                    orb_return = orbfex.main(pfad, orb_obj)
                    

                    # speichern in Kunstwerke Tabelle
                    sql = "INSERT INTO kunstwerke(bild_zahl, titel, name, jahr, link, ort, material, genre, stil, preis, galerie_name, hash, ahash, orb) VALUES (?, ?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    val = [
                    (bild_zahl, titel, name, jahr, link, ort, material, genre, stil, preis, galerie_name, hash_return, ahash_return, orb_return)
                    ]   
                    c.executemany(sql, val)
yield
                except Exception as e:
                    print(e)
                
    # Speichern der Änderungen und Schließen der Verbindung
    conn.commit()
    conn.close()

def main():
    erstellen()
    fuellen()

if __name__ == "__main__":
    main()