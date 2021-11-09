#
# convert JSON mess to SQL dataset
# only works for Lucas Davis's scraper
#

# TODO cleanup dataset (remove NONE)

import json
import sqlite3
from pathlib import Path
import re

# connect
DBNAME = "WikiartDatasetMultiStyles.db"
DSPATH = "D:/wikiartbackup/saved/" # path of the "saved" folder

conn = sqlite3.connect(DBNAME)
c = conn.cursor()

def create():
    # create artist Table
    c.execute("""CREATE TABLE IF NOT EXISTS artists (
                id integer PRIMARY KEY AUTOINCREMENT,
                name text NOT NULL,
                birthdate text,
                deathdate text,
                wikipedia text
                )""")
    # create artwork Table
    c.execute("""CREATE TABLE IF NOT EXISTS artworks (
                id integer PRIMARY KEY AUTOINCREMENT,
                imgid integer NOT NULL,
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
        print(artistcounter)
        for artwork in artworkData:
            counter += 1
            # print("image", counter, "|", len(artworkData), "  artist", artistcounter, "|", len(jsonFile))
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
                    if None.__ne__(artwork["style"]):

                        # Informationen werden für jedes Bild gesammelt
                        imgid = artwork["contentId"]
                        artworktitle = artwork["title"]
                        artistName = artwork["artistName"]
                        year = artwork["completitionYear"]
                        link = artwork["image"]
                        location = artwork["location"]
                        genre = artwork["genre"]
                        
                        # style can have multiple attributes.
                        # To fix this we replace spaces in each label with "_"
                        # multiple labels are seperated by ", " we replace this with just a ","
                        # some style labels already contain "-" so we replace this with "_" might help down the road
                        # old: Art Nouveau (Modern), Biedermeier
                        # new: Art_Nouveau_(Modern),Biedermeier
                        # print("old:", style)
                        style = artwork["style"]
                        style = re.sub(", ", ",", style)
                        style = re.sub(" ", "_", style)
                        style = re.sub("-", "_", style)
                        style = style.split(",")
                        # print("new:", style)
                        galleryName = artwork["galleryName"]
                        description = artwork["description"]

                        # create new row for every style that an image has
                        for singlestyle in style:
                            createrow(imgid, artworktitle, artistName, path, year, link, location, genre, singlestyle, galleryName, description)


                    else:
                        pass
                        # print(artwork["genre"], "genre missing")

                else:
                    # print(path, "image missing")
                    pass


            except Exception as e:
                print(e)

    conn.commit()
    conn.close()

# create a row of the database # TODO ugly
def createrow(imgid, artworktitle, artistName, path, year, link, location, genre, singlestyle, galleryName, description):

    sql = "INSERT INTO artworks(imgid, title, artistName, path, year, link, location, genre, style, galleryName, description) VALUES (?,?,?,?, ?, ?, ?, ?, ?,?,?)"
    val = [(imgid, artworktitle, artistName, path, year, link, location, genre, singlestyle, galleryName, description)]
    c.executemany(sql, val)
    

if __name__ == "__main__":
    create()
    fill()