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

    outfolder = "./visualisations_out/"

    # might work with batching
    images, labels = tuple(zip(*test_ds)) 

    # print(tf.shape(images))


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

    def _plot_save_cfm(conf_matrix, annotation=''):
        # conf_matrix = confusion.numpy()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s="{:.2f}".format(conf_matrix[i, j]) , va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix' + annotation, fontsize=18)
        # plt.show()
        plt.savefig(outfolder + plottitle + '_cm' + annotation + '.png', dpi=500)
    
    conf_matrix = confusion.numpy()
    _plot_save_cfm(conf_matrix=conf_matrix)
    
    row_sums = conf_matrix.sum(axis=1)
    norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]
    _plot_save_cfm(conf_matrix=norm_conf_matrix, annotation='_Normalized')


    # plot history
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.title(plottitle + ' Loss/Accuracy', fontsize=18)
    plt.savefig(outfolder + plottitle + '_history.png', dpi=500)
    print("saved plotting results for", plottitle)


if __name__ == "__main__":
    print("please dont directly execute this script")