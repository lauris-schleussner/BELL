# resize images using inpainting to improve padding
# https://stackoverflow.com/questions/52735231/how-to-select-all-non-black-pixels-in-a-numpy-array/52737768

from bellutils.getfrompath import getfrompath
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

imglist = getfrompath("D:/BELL/dataset_resized")

for path in tqdm(imglist):

    print(path)
    image = plt.imread(path)
    black_pixels_mask = np.all(image == [0, 0, 0], axis=-1)
    plt.imshow(black_pixels_mask)
    plt.show()
    quit()

