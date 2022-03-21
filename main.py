import train_cnn
import train_vgg16
import cfm_history
import train_alexnet
import train_autoencoder
import train_resnet50
import train_xception


# first network
print("training cnn1")
cfm_history.main(train_cnn.main( EPOCHS = 1), plottitle="CNN 1")

# alexnet
print("training alexnet")
cfm_history.main(train_alexnet.main(EPOCHS = 1), plottitle="AlexNet")

# vgg16 no weights
print("training vgg16 without weights")
cfm_history.main(train_vgg16.main(EPOCHS = 1, pretrained = None), plottitle="VGG16 newly trained")

# vgg16 with weights
print("training vgg16 with pretrained weights")
cfm_history.main(train_vgg16.main(EPOCHS = 1, pretrained = "imagenet"), plottitle="VGG16 pretrained")

# resnet no weights
print("training resnet50 without weights")
cfm_history.main(train_resnet50.main(EPOCHS = 1, pretrained = None), plottitle="ResNet50 newly trained")

# resnet pretrained weights
print("training resnet50 with pretrained weigths")
cfm_history.main(train_resnet50.main(EPOCHS = 1, pretrained = "imagenet"), plottitle="ResNet50 pretrained")

# xception new weights
print("training xception without weigths")
cfm_history.main(train_xception.main(EPOCHS = 1, pretrained = None), plottitle="Xception newly trained")

# xception pretrained weights
print("training xception with pretrained weigths")
cfm_history.main(train_xception.main(EPOCHS = 1, pretrained = "imagenet"), plottitle="Xception pretrained")

# autoencoder, no visuals
print("training encoder")
train_autoencoder.main(EPOCHS = 1)