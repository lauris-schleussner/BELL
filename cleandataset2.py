import os
from PIL import Image
import glob
import sys

imglist = glob.glob("wikiart-master\saved\images\*\*\*.jpg")
print(len(imglist))

for filename in imglist:
    with Image.open(filename) as img:
        width, height = img.size

    if width <= 2000 or height <= 2000:
        print(filename, width, height)
        img.thumbnail([sys.maxsize, 100], Image.ANTIALIAS)
        img.save(filename)

