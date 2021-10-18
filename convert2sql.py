#
# convert JSON mess to SQL dataset
# only works for Lucas Davis's scraper
#

# TODO cleanup dataset (remove NONE)

import json
import sqlite3
from pathlib import Path

# connect
DBNAME = "cleanedWikiartDataset.db"
DSPATH = "wikiart-master/saved/" # path of the "saved" folder 
conn = sqlite3.connect(DBNAME)
c = conn.cursor()

def create():
    # create artist Table
    print("asjdkh")
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
                genre text,
                style text,
                galleryName text,
                description text
                )""")

def fill():
    # artists.json contains a list of all artists
    with open(DSPATH + "meta/artists.json", encoding="latin-1") as f:
        jsonFile = json.load(f)

    # obtain values and loop over every artist
    artistcounter = 0
    for artistDict in jsonFile:
        artistcounter += 1

        artistUrl = artistDict["url"]
        name = artistDict["artistName"]
        birthdate = artistDict["birthDayAsString"]
        deathdate = artistDict["deathDayAsString"]
        wikipedia = artistDict["wikipediaUrl"]

        # write data to artist table
        sql = "INSERT INTO artists(name, birthdate, deathdate, wikipedia) VALUES (?, ?, ?, ?)"
        val = [
            (name, birthdate, deathdate, wikipedia)
        ]
        c.executemany(sql, val)

        # each artist's own JSON file is opened
        # a list of all their artworks is loaded
        try:
            with open(DSPATH + "meta/" + artistUrl + ".json", encoding="latin-1") as f:
                artworkData = json.load(f)

        except Exception as e:
            print(e)
            print(name, "missing")
            continue

        # open each 
        counter = 0
        for artwork in artworkData:
            counter += 1
            print("image", counter, "|", len(artworkData), "  artist", artistcounter, "|", len(jsonFile))
            try:
 
                # path of each artwork is combined 
                # "wikiart-master/saved/images/-artistUrl-/-year-/-contentId-"
                # year "0" converted to "unknown-year"
                if len(artwork["yearAsString"]) == 0:
                    year = "unknown-year"
                else:
                    year = artwork["yearAsString"]
                path = DSPATH + "images/" + str(artwork["artistUrl"]) + "/" + year +"/" + str(artwork["contentId"]) + ".jpg"

                # check if image exists, skip if not
                if Path(path).exists():
                    if artwork["genre"] is not None:


                        # Informationen werden für jedes Bild gesammelt
                        imgid = artwork["contentId"]
                        artworktitle = artwork["title"]
                        artistName = artwork["artistName"]
                        year = artwork["completitionYear"]
                        link = artwork["image"]
                        location = artwork["location"]
                        genre = artwork["genre"]
                        style = artwork["style"]
                        galleryName = artwork["galleryName"]
                        description = artwork["description"]

                        # save in table "artworks"
                        sql = "INSERT INTO artworks(id, title, artistName, path, year, link, location, genre, style, galleryName, description) VALUES (?, ?,?,?, ?, ?, ?, ?, ?,?,?)"
                        val = [(imgid, artworktitle, artistName, path, year, link, location, genre, style, galleryName, description)]
                        c.executemany(sql, val)
                        
                    else:
                        print(artwork["genre"], "genre missing")

                else:
                    # print(path, "image missing")
                    pass


            except Exception as e:
                print(e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create()
    fill()