import os
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import math
from sklearn.cluster import KMeans
import tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(threshold=1000)

def KMeanfnc(numclusters):
    clf = KMeans(n_clusters=numclusters)
    clf.fit(DataSet)

    centroids = clf.cluster_centers_
    labels = clf.labels_

    print(centroids)
    print(labels)
    np.savetxt('100x100Labels.txt', labels, fmt='%i', delimiter=',')

def AutoEncoder():
    learning_rate = 0.01
    training_epochs = 100
    batch_size = 200
    examples_to_show = 20

    n_hidden_1 = int(n_features / 5)
    n_hidden_2 = int(n_features / 5)
    n_hidden_3 = int(n_features / 5)
    n_hidden_4 = int(n_features / 5)


    X = tf.placeholder("float", [None, n_features])

    weights = {'encoder_h1': tf.Variable(tf.random_normal([n_features, n_hidden_1])),
               'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
               'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
               'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
               'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
               'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
               'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
               'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_features]))}

    biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
              'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
              'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
              'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
              'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
              'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
              'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
              'decoder_b4': tf.Variable(tf.random_normal([n_features]))}

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    encoder_op = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op, weights['decoder_h1']), biases['decoder_b1']))
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    cost = tf.reduce_mean(tf.pow(X - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            epoch_loss, i = 0, 0
            while i < lengthdata:
                start = i
                end = i + batch_size
                batch_x = np.array(DataSet[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
                epoch_loss += c
                i += batch_size
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

        encode_decode = sess.run(y_pred, feed_dict={X: DataSet[:examples_to_show]})

        print(encode_decode)
        print(encode_decode.shape)
        print(type(encode_decode))

        f, a = plt.subplots(2, 20, figsize=(20, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(DataSet[i], (math.sqrt(n_features), math.sqrt(n_features))))
            a[1][i].imshow(np.reshape(encode_decode[i], (math.sqrt(n_features), math.sqrt(n_features))))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
        sess.close()
        tf.reset_default_graph()

    return

def ModelSelect():
    #ans = input('\nTrain which model? (kmeans, autoencoder, exit): ')
    ans = 'autoencoder'
    if ans in ['kmeans']:
        numclusters = int(input('\nHow many clusters? '))
        KMeanfnc(numclusters)
    elif ans in ['autoencoder']:
        AutoEncoder()
    elif ans in ['exit']: exit()
    else:
        print('Improper Selection')
        ModelSelect()
        return

def LoadData():
    dirs = os.listdir("unsuperviseddata/")
    print("\nAvailable Datasets: \n")
    for file in dirs: print(file)
    #datarun = input("\nEnter a Dataset: ")
    #datarun = '1chargingbehaviour101308UnsupervisedData50x50.txt'
    datarun = 'toydata50x50.txt'
    return datarun

def CheckDataName():
    while True:
        datarun = LoadData()
        dataname = datarun[:-4]
        try:
            DataSet = np.loadtxt('unsuperviseddata/{}'.format(datarun), delimiter=",")
            return DataSet, dataname
            break
        except FileNotFoundError:
            try:
                DataSet, dataname = np.loadtxt('unsuperviseddata/{}.txt'.format(datarun), delimiter=","), datarun
                return DataSet, dataname
                break
            except FileNotFoundError:
                try:
                    DataSet, dataname = np.loadtxt('unsuperviseddata/{}.csv'.format(datarun), delimiter=","), datarun
                    return DataSet, dataname
                    break
                except FileNotFoundError:
                    print('Dataset Not Found')

DataSet, dataname = CheckDataName()
lengthdata = len(DataSet)
n_features = len(DataSet[0])

ModelSelect()

while True:
    ans = input('Run another model? (y/n) ')
    if ans in ['y']: ModelSelect()
    else: exit()