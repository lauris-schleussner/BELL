# check the distribution of multiple-labeled images
# some images in the original dataset can have multiple style attributes
# we want to check if this influences the quality of the dataset

import sqlite3

DBNAME = "WikiartDatasetMultiStyles.db"
DSPATH = "D:/wikiartbackup/saved/" # path of the "saved" folder

conn = sqlite3.connect(DBNAME)
c = conn.cursor()

# print and save a list of each style and how many images they contain
def listall():
    # get list of all unique styles returns list of tupels with one datapair each
    c.execute("SELECT DISTINCT style FROM artworks")
    res = c.fetchall()
    print(res)

    orderlist = [] # [genre, number of images]
    for genre in res:
        print(genre)
        c.execute("SELECT COUNT(style) FROM artworks WHERE style = '" + str(genre[0]) + "'")
        res = c.fetchone() 
        orderlist.append([genre, res[0]])

    orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

    textfile = open("singlelabeldist.txt", "w")
    for element in orderlist:
        string = (str(element[0][0]) + " " + str(element[1]) + "\n").encode('ascii', 'ignore').decode('ascii')
        textfile.write(string)
    textfile.close()

    print("###################################################################################")
    print(orderlist)

    # sort result list by number of images each genre has
    orderlist = sorted(orderlist, key=lambda l:l[1], reverse=True)

    print(orderlist)

    return orderlist

def showpairs():

    
    # create list that holds every imageid and their associated styles
    stylelist = []

    # select style and imageid from every image
    sql = 'SELECT style, imgid FROM artworks'
    c.execute(sql)
    res = c.fetchall()

    # to count: every imgid is associated with all their styles
    styledict = {}
    for style, id in res:

        try: # if entry for id already exists, append new style
            styledict[id].append(style)
        except: # if not, create entry of that style 
            styledict[id] = [style]
            
    # iterate over all styles and append them to a 2d list [[a,b,c], [a,c], [b,c], ...] if they are at least 2
    for styles in styledict.values():
        if len(styles) != 1:
            stylelist.append(styles)
    # https://stackoverflow.com/questions/27733685/iterating-over-dict-values/27733728
    # find most common combinations
    from collections import Counter
    from itertools import combinations

    d  = Counter()
    for sub in stylelist:
        if len(stylelist) < 2:
            continue
        sub.sort()
        for comb in combinations(sub,2):
            d[comb] += 1

    res = d.most_common()
    for i in res:
        print(i)

    

if __name__ == "__main__":
    listall()