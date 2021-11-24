# return list of all images in a directory
import glob

def getfrompath(path):
    imglist = glob.glob(path + "\**\*.jpg", recursive=True)
    return imglist


if __name__ == "__main__":
    main("D:\BELL\wikiart-master\saved\images")

