import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception

import os
import distutils.dir_util


AE_PARAM_EPOCHS = 16
PARAM_WAB = True
TAGS = ['244px']

os.system('rm -r models')
os.system('rm -r visualisations_out')

if not os.path.isdir("autoenc/resized"):
    print("copying './resized' to './autoenc/resized' ")
    distutils.dir_util.copy_tree("./resized", "./autoenc/resized")
    # distutils.dir_util.copy_tree("./resized100", "./autoenc/resized")
    print("copy done")

if not os.path.isdir("visualisations_out"):
    print("creating visualisation directory")
    os.mkdir("visualisations_out")

print("training encoder")
train_autoencoder.main(EPOCHS=AE_PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS)

os.system('rm -r ./autoenc/resized')