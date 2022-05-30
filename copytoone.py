from bellutils.getfrompath import getfrompath
import shutil
from tqdm import tqdm


pathlist = getfrompath("wikiart-master/saved/")

for path in tqdm(pathlist):
    shutil.copy(path,"E:/BELL/originalsize")
