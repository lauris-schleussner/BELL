# return list of all imagepaths in a directory, recursively depth doesnt matter
# takes some time on first execution, probably windows caching
import glob

def getfrompath(path):
    imglist = glob.glob(path + "\**\*.jpg", recursive=True)
    return imglist


if __name__ == "__main__":
    getfrompath("D:\BELL\wikiart-master\saved\images")

