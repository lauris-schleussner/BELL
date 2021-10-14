#
# convert JSON mess to SQL dataset
# only works for Lucas Davis's scraper
#

import json
import sqlite3
from pathlib import Path

# connect
DBNAME = "wikiartDataset.db"
DSPATH = "wikiart-master/saved/meta/" # path of the "meta" folder 
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

def create():
    # create artist Table
    c.execute("""CREATE TABLE IF NOT EXISTS artists (
                id integer PRIMARY KEY,
                name text NOT NULL,
                birthdate text,
                deathdate text,
                wikipedia text
                )""")
    # create artwork Table
    c.execute("""CREATE TABLE IF NOT EXISTS artworks (
                id integer PRIMARY KEY,
                title text NOT NULL,
                artistName text,
                path text,
                year text,
                link text,
                location text,
                material text,
                genre text,
                style text,
                worth real,
                galleryName text,
                description text
                )""")

def fill():
    # artists.json contains a list of all artists
    with open(DSPATH + "artists.json", encoding="latin-1") as f:
        jsonFile = json.load(f)

    # obtain values and loop over every artist
    for artistDict in jsonFile:
        artistUrl = artistDict["url"]
        name = artistDict["artistName"]
        birthdate = artistDict["birthDayAsString"]
        deathdate = artistDict["deathDayAsString"]
        wikipedia = artistDict["wikipediaUrl"]

        # write data to artist table
        sql = "INSERT INTO artist(name, birtdate, deathdate, wikipedia) VALUES (?, ?, ?, ?)"
        val = [
            (name, birthdate, deathdate, wikipedia)
        ]
        c.executemany(sql, val)

        # each artist's own JSON file is opened
        # a list of all their artworks is loaded
        try:
            with open(DSPATH + artistUrl + ".json", encoding="latin-1") as f:
                artworkData = json.load(f)

        except Exception as e:
            print(e)
            print(name, "missing")
            continue

        # open each 
        for artwork in artworkData:

            try:
 
                # path of each artwork is combined 
                # "wikiart-master/saved/images/-artistUrl-/-year-/-contentId-"
                # year "0" converted to "unknown-year"
                if len(artwork["yearAsString"]) == 0:
                    year = "unknown-year"
                else:
                    year = artwork["yearAsString"]
                path = DSPATH + str(artwork["artistUrl"]) + "/" + year +"/" + str(artwork["contentId"]) + ".jpg"

                # check if image exists, skip if not
                if Path(path).exists():


                    # Informationen werden für jedes Bild gesammelt
                    imgid = artwork["contentId"]
                    title = artwork["title"]
                    artistName = artwork["artistName"]
                    year = artwork["completitionYear"]
                    link = artwork["image"]
                    location = artwork["location"]
                    material = artwork["material"]
                    genre = artwork["genre"]
                    style = artwork["style"]
                    worth = artwork["lastPrice"]
                    galleryName = artwork["galleryName"]
                    description = artwork["description"]

                    # save in table "artworks"
                    sql = "INSERT INTO kunstwerke(id title artistName path year link location material genre style worth galleryName description) VALUES (?, ?,?,?,?, ?, ?, ?, ?, ?, ?,?,?)"
                    val = [(imgid, title, artistName, path, year, link, location, material, genre, style, worth, galleryName, description)]
                    c.executemany(sql, val)
                else:
                    print(path, "image missing")


            except Exception as e:
                print(e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create()
    fill()