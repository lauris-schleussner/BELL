# create plot of all the different image sizes

import glob
import imagesize
import numpy as np
import matplotlib.pyplot as plt
from pymage_size import get_image_size


imglist = glob.glob("wikiart-master\saved\images\*\*\*.jpg")
print(len(imglist))

wlist = []
hlist = []

counter = 0
for path in imglist:
    
    counter += 1

    if counter % 1000 == 0:
        print(counter)


    img_format = get_image_size(path)
    w, h = img_format.get_dimensions()

    wlist.append(w)
    hlist.append(h)

plt.scatter(wlist, hlist)
plt.show()