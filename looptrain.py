import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception

import os
import distutils.dir_util


PARAM_EPOCHS = 30
PARAM_WAB = True
GROUP_RUN_TAG = 'looprun03'
TAGS = [GROUP_RUN_TAG]

for i in range(10):
    # first network
    print("training cnn1")
    train_cnn.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS, savemodel=False)

    # alexnet
    print("training alexnet")
    train_alexnet.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS, savemodel=False)

    # vgg16 no weights
    print("training vgg16 without weights")
    train_vgg16.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=False, add_tags=TAGS, savemodel=False)