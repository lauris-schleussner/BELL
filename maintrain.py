import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception

import os
import distutils.dir_util


if not os.path.isdir("autoenc/resized"):
    print("copying './resized' to './autoenc/resized' ")
    distutils.dir_util.copy_tree("./resized", "./autoenc/resized")
    print("copy done")

# first network
print("training cnn1")
cfm_history.main(train_cnn.main(EPOCHS = 100), plottitle="CNN 1")

# alexnet
print("training alexnet")
cfm_history.main(train_alexnet.main(EPOCHS = 100), plottitle="AlexNet")

# vgg16 no weights
print("training vgg16 without weights")
cfm_history.main(train_vgg16.main(EPOCHS = 300, pretrained = False), plottitle="VGG16 newly trained")

# vgg16 with weights
print("training vgg16 with pretrained weights")
cfm_history.main(train_vgg16.main(EPOCHS = 300, pretrained = True), plottitle="VGG16 pretrained")

# resnet no weights
print("training resnet50 without weights")
cfm_history.main(train_resnet50.main(EPOCHS = 300, pretrained = False), plottitle="ResNet50 newly trained")

# resnet pretrained weights
print("training resnet50 with pretrained weigths")
cfm_history.main(train_resnet50.main(EPOCHS = 300, pretrained = True), plottitle="ResNet50 pretrained")

# xception new weights
print("training xception without weigths")
cfm_history.main(train_xception.main(EPOCHS = 300, pretrained = False), plottitle="Xception newly trained")

# xception pretrained weights
print("training xception with pretrained weigths")
cfm_history.main(train_xception.main(EPOCHS = 300, pretrained = True), plottitle="Xception pretrained")

print("training encoder")
train_autoencoder.main(EPOCHS = 400)