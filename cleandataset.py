# https://stackoverflow.com/questions/46854496/python-script-to-detect-broken-images
import glob
from PIL import Image
import os

imglist = glob.glob("wikiart-master/saved/images/*/*/*.jpg")
print(len(imglist))

removelist = []
counter = 0
for filename in imglist:
    counter +=1
    print(counter)
    if filename.endswith('.jpg'):
        try:
            img = Image.open(filename)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print(e)
            print(filename)
            removelist.append(filename)
            
print(removelist)
failedlist = []

for file in removelist:
    print(file)
    try:
        os.remove(filename)
    except:
        print("failed to remove", file)
        failedlist.append(file)

print("failed:")
print(failedlist)