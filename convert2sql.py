#
# convert JSON mess to SQL dataset
# only works for Lucas Davis's scraper
#

# TODO cleanup dataset (remove NONE)

import json
import sqlite3
from pathlib import Path
import re
from tqdm import tqdm
import os
import plac


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
                filename text,
                path text,
                year text,
                link text,
                location text,
                genre text,
                style text,
                galleryName text,
                description text,
                corrupt bool,
                used bool
                )""")

def fill(DSPATH):
    # artists.json contains a list of all artists
    with open(DSPATH + "meta/artists.json", encoding="latin-1") as f:
        jsonFile = json.load(f)

    # obtain values and loop over every artist
    for artistDict in tqdm(jsonFile):

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
        
        for artwork in artworkData:
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
                        # adjust path prefix here
                        year = artwork["completitionYear"]
                        link = artwork["image"]
                        location = artwork["location"]
                        genre = artwork["genre"]
                        filename = os.path.basename(path)
                        path = path
                        
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

                        relevantstyles = ["Impressionism", "Realism", "Romanticism", "Expressionism" "Art_Nouveau_(Modern)"]

                        # if the style is part of the "relevant ones" and if an image only has one style attribute. the "used" flag is set
                        # images with multiple styles are not stored in the db
                        if len(style) == 1:
                            if str(style[0]) in relevantstyles:
                                used = True
                            else:
                                used = False

                        # corrupt flag is not set by default as it is set later in the cleaning process
                        corrupt = False

                        createrow(imgid, artworktitle, artistName, filename, path, year, link, location, genre, style[0], galleryName, description, corrupt, used)      
                        
                        '''
                        # create new row for every style that an image has
                        for singlestyle in style:
                            createrow(imgid, artworktitle, artistName, filename, year, link, location, genre, singlestyle, galleryName, description)
                        '''
                    
                    else:
                        pass
                        # print(artwork["genre"], "genre missing")

                else:
                    # print(path, "image missing")
                    pass

        
            except Exception as e:
                print(e)
    

# create a row of the database # TODO ugly
def createrow(imgid, artworktitle, artistName, filename, path, year, link, location, genre, singlestyle, galleryName, description, corrupt, used):

    sql = "INSERT INTO artworks (imgid, title, artistName, filename, path, year, link, location, genre, style, galleryName, description, corrupt, used) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    val = [(imgid, artworktitle, artistName, filename, year, link, path, location, genre, singlestyle, galleryName, description, corrupt, used)]
    c.executemany(sql, val)

def main(DSPATH = "wikiart-master/saved/", DBNAME = "database.db"):
    "Path of the 'saved' Folder (example: 'D:/BELL/data/saved/'), name of the datbase (example: 'mybeautifulnewsqldataset.db')"
    """
    python convert2sql.py wikiart-master/saved/ database.db
    """

    # connect to DB
    global conn, c
    conn = sqlite3.connect(DBNAME)
    c = conn.cursor()

    # create database layout
    create()

    # fill database
    fill(DSPATH)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    plac.call(main)