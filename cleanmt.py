# takes in the folder location of the "saved" folder from the original download
# turns it into 2 folders:
#       - originalsize
#       - resized
# both folders contain cleaned up versions of the dataset, compressed into a single 1d folder
# same as clean.py but with multiprocessing
# TODO would have been smarter to not use Threadpooling, still works
# uses max number of cores by default!
# there will be a few warnings, but the images are mostly fine

import os
from bellutils.getfrompath import getfrompath
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence Tensorflow
from tqdm import tqdm
import tensorflow as tf
import plac
import logging
logging.basicConfig(filename='errorsfromcleaning.log', encoding='utf-8', level=logging.DEBUG,  format='%(asctime)s %(levelname)-8s %(message)s',  datefmt='%Y-%m-%d %H:%M:%S')
from shutil import copyfile
from pathlib import Path
import sqlite3
import numpy as np

import multiprocessing as mp

def findcorruptresizeandcopy(arguments):

    # retrieve arguments, because map() wont work with multiple
    pathlist = arguments[0]
    outfolderresized = arguments[1]

    corruptimages = []
    validimages = []
    for imgpath in tqdm(pathlist):
        
        # get filename
        name = os.path.basename(imgpath)
        try:

            # open and read file
            image_o = tf.io.read_file(imgpath)
            image_o = tf.io.decode_image(image_o, channels=3)

            # validity check, try to resize
            image_test = tf.image.convert_image_dtype(image_o, tf.float32)
            image_test = tf.image.resize_with_pad(image_test, 500, 500)

            # save
            tf.keras.utils.save_img(outfolderresized + name, image_test)

            # if passes as valid add to list
            validimages.append(imgpath)

        except Exception as e:
            # see why images were corrupted
            logging.exception("corrupt file found during file checking: " + str(imgpath))
            corruptimages.append(imgpath)

    return [validimages, corruptimages]

def copytofolder(validimagepaths, outfolderoriginal):

    for frompath in tqdm(validimagepaths):
        filename = os.path.basename(frompath)
        topath = outfolderoriginal + filename
        copyfile(frompath, topath)

    return 0

def removefromdb(corruptimages, dbname):

    conn = sqlite3.connect(dbname)
    c = conn.cursor()

    # get id of the corrupt image from filepath
    for filename in tqdm(corruptimages):
        p = Path(filename)
        id = p.stem

        c.execute("UPDATE artworks SET corrupt = True WHERE imgid = '" + id + "'")

    conn.commit()
    
    return 0
    
def main(inpath = "wikiart-master/saved/", outfolderoriginal = "originalsize/", outfolderresized = "resized/", dbname = "database.db", corruptlogfile = "corrupt.txt", cores = mp.cpu_count()):

    # 0 init multiprocessing and create relevant directories or dependancy code fails
    print("number of cores used:" , cores)
    pool = mp.Pool(cores)

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

    # 3. All invalid images are removed from the Database
    print("removing from database (singlethreaded)")
    removefromdb(corruptimages, dbname)

if __name__ == "__main__":
    plac.call(main)