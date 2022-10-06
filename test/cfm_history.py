# plot history
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
import matplotlib.pyplot as plt
import pandas as pd

def main(traindata, plottitle):

    model = traindata[0]
    history = traindata[1]
    test_ds = traindata[2]

    outfolder = "E:/BELL/plots/"

    # plot history
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.savefig(outfolder + plottitle + '_history.png', dpi=500)
    print("saved plotting results for", plottitle)


    """




    # might work with batching
    images, labels = tuple(zip(*test_ds)) 

    print(tf.shape(images))
    quit()

    # build one batch with all test pictures 
    # Final tensor format (Number of Pictures, IMGSIZE, IMGSIZE, 3)
    images = np.array(images)
    labels = np.array(labels)
    labels = labels.flatten()
    # print("labels: ", labels)

    rawpredictions = model.predict(images)

    argmaxpredictions = tf.argmax(rawpredictions, axis=-1)
    argmaxpredictions = argmaxpredictions.numpy()
    argmaxpredictions = argmaxpredictions.flatten()
    # print("argmaxpredictions:", argmaxpredictions)

    weights = []

    for i in range(len(rawpredictions)):
        rawsingleprediction = rawpredictions[i]
        argmaxindex = argmaxpredictions[i]
        weights.append(rawsingleprediction[argmaxindex])


    confusion = tf.math.confusion_matrix(labels=labels, predictions=argmaxpredictions)
    # print(confusion.shape)

    conf_matrix = confusion.numpy()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    plt.savefig(outfolder + plottitle + '_cm.png', dpi=500)


    # plot history
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.savefig(outfolder + plottitle + '_history.png', dpi=500)
    print("saved plotting results for", plottitle)

 """
if __name__ == "__main__":
    print("please dont directly execute this script")