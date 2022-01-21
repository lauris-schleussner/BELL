# takes in the folder location of the "saved" folder from the original download
# turns it into 3 folders:
#       - originalsize
#       - resized
# both folders contain cleaned up versions of the dataset in only one folder
# same as clean.py but with multiprocessing
# would have been smarter to not use Threadpooling, still works
# uses max number of cores by default

import os
from bellutils.getfrompath import getfrompath
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
from tqdm import tqdm
import tensorflow as tf
import plac
import logging
import traceback
logging.basicConfig(filename='errors.log', encoding='utf-8', level=logging.DEBUG,  format='%(asctime)s %(levelname)-8s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S')
from shutil import copyfile
from pathlib import Path
import sqlite3
import numpy as np
import pickle

import multiprocessing as mp

def findcorruptresizeandcopy(arguments):

    pathlist = arguments[0]
    outfolderresized = arguments[1]

    corruptimages = []
    validimages = []
    for imgpath in tqdm(pathlist):
        
        name = os.path.basename(imgpath)
        try:

            # open and read file
            image_o = tf.io.read_file(imgpath)
            image_o = tf.io.decode_image(image_o, channels=3)

            # validity check
            image_test = tf.image.convert_image_dtype(image_o, tf.float32)
            image_test = tf.image.resize_with_pad(image_test, 500, 500)

            # save
            tf.keras.utils.save_img(outfolderresized + name, image_test)

            # if passes as valid add to list
            validimages.append(imgpath)

        except Exception as e:
            # logging.exception("corrupt file found during file checking: " + str(imgpath))
            corruptimages.append(imgpath)

    return [validimages, corruptimages]

def copytofolder(validimagepaths, outfolderoriginal):

    for frompath in tqdm(validimagepaths):
        filename = os.path.basename(frompath)
        topath = outfolderoriginal + filename
        copyfile(frompath, topath)

    return 0

def resizeandcopy(arguments):
    validimagepaths = arguments[0]
    outfolderresized = arguments[1]

    # for images that are found to be broken during saving
    corruptimages = []

    validimages = []
    for imgpath in tqdm(validimagepaths):
        
        # still fails sometimes
        try:

            name = os.path.basename(imgpath)
            
            # resize file ans save to output folder
            image = tf.io.read_file(imgpath)
            image = tf.io.decode_image(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, 500, 500)
            tf.keras.utils.save_img(outfolderresized + name, image)

        except Exception as e:
            # logging.exception("corrupt file found during resizing: " + str(imgpath))
            corruptimages.append(imgpath)

        
    return corruptimages

def removefromdb(corruptimages, dbname):

    conn = sqlite3.connect(dbname)
    c = conn.cursor()

    # get id of the corrupt image from filepath
    for filename in tqdm(corruptimages):
        print(filename)
        p = Path(filename)
        id = p.stem
        print(id)

        c.execute("UPDATE artworks SET corrupt = True WHERE imgid = '" + id + "'")

    conn.commit()
    return 0
    
def main(inpath = "wikiart-master/saved/", outfolderoriginal = "originalsize/", outfolderresized = "resized/", dbname = "database.db", corruptlogfile = "corrupt.txt", cores = mp.cpu_count()):

    # -1 init multiprocessing
    print("number of cores used:" , cores)
    pool = mp.Pool(cores)

    # 0. create relevant directories or dependancy code fails
    try:
        os.mkdir(outfolderresized)
        os.mkdir(outfolderoriginal)
        print("created directories: ", outfolderoriginal, outfolderresized)
    except:
        print("could not create directories, do they already exist?")
        

    # 1. all images are located and added to a list. Each image path gets opened and checked for valid files
    # because this process in mostly CPU bound it can be multithreaded
    print("checking images (multithreaded)")

    # list is first split into chunks. Each thread is then assigned 
    pathlist = getfrompath(inpath) 
    if len(pathlist) == 0:
        print("WARNING, no images found in", inpath )
    pathlist_split = np.array_split(pathlist, cores)

    # requires iterable
    arguments = []
    for i in pathlist_split:
        arguments.append([i, outfolderresized])
    fullres = pool.map(findcorruptresizeandcopy, arguments) # get full operation result, probably not the right usage of a Threadpool, works anyways

    # because map() returns a split list, the results have to be assembled into one list
    validimagepaths = []
    corruptimages = []
    for threadres in fullres:
        for valid in threadres[0]:
            validimagepaths.append(valid)
        for corrupt in threadres[1]:
            corruptimages.append(corrupt)

    # 2. all valid images are copied to a folder
    print("originalsize copy process to folder (singlethreaded)")
    copytofolder(validimagepaths, outfolderoriginal)

    # 4. All invalid images are removed from the Database
    # 4.1 All invalid images are logged to a file for removal in case of recreating the database
    print("removing from database (singlethreaded)")
    removefromdb(corruptimages, dbname)

if __name__ == "__main__":
    plac.call(main)