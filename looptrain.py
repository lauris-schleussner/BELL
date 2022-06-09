import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception

import os
import distutils.dir_util


PARAM_EPOCHS = 50
PARAM_WAB = True
GROUP_RUN_TAG = 'looprun01'
TAGS = [GROUP_RUN_TAG]

# first network
print("training cnn1")
train_cnn.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS)

# alexnet
print("training alexnet")
train_alexnet.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS)

# vgg16 no weights
print("training vgg16 without weights")
train_vgg16.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=False, add_tags=TAGS)