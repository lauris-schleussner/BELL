import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception

import os
import distutils.dir_util


PARAM_EPOCHS = 100 # 200
AE_PARAM_EPOCHS = 3
PARAM_WAB = True
GROUP_RUN_TAG = 'comprun06'
TAGS = [GROUP_RUN_TAG, 'olddb', '244px', 'tuned_lr']

os.system('rm -r models')
os.system('rm -r visualisations_out')

# if not os.path.isdir("autoenc/resized"):
#     print("copying './resized' to './autoenc/resized' ")
#     distutils.dir_util.copy_tree("./resized", "./autoenc/resized")
#     print("copy done")

if not os.path.isdir("visualisations_out"):
    print("creating visualisation directory")
    os.mkdir("visualisations_out")

# first network
print("training cnn1")
cfm_history.main(train_cnn.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS), plottitle="CNN 1")

# alexnet
print("training alexnet")
cfm_history.main(train_alexnet.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, add_tags=TAGS), plottitle="AlexNet")

# vgg16 no weights
print("training vgg16 without weights")
cfm_history.main(train_vgg16.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=False, add_tags=TAGS), plottitle="VGG16 newly trained")

# vgg16 with weights
print("training vgg16 with pretrained weights")
cfm_history.main(train_vgg16.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=True, add_tags=TAGS), plottitle="VGG16 pretrained")

# resnet no weights
print("training resnet50 without weights")
cfm_history.main(train_resnet50.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=False, add_tags=TAGS), plottitle="ResNet50 newly trained")

# resnet pretrained weights
print("training resnet50 with pretrained weigths")
cfm_history.main(train_resnet50.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=True, add_tags=TAGS), plottitle="ResNet50 pretrained")

# xception new weights
print("training xception without weigths")
cfm_history.main(train_xception.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=False, add_tags=TAGS), plottitle="Xception newly trained")

# xception pretrained weights
print("training xception with pretrained weigths")
cfm_history.main(train_xception.main(EPOCHS=PARAM_EPOCHS, WAB_FLAG=PARAM_WAB, pretrained=True, add_tags=TAGS), plottitle="Xception pretrained")

# print("training encoder")
# train_autoencoder.main(EPOCHS = AE_PARAM_EPOCHS)

os.system('tar -zcvf '+GROUP_RUN_TAG+'_models.tar.gz models/')
os.system('tar -zcvf '+GROUP_RUN_TAG+'_viz.tar.gz visualisations_out/')