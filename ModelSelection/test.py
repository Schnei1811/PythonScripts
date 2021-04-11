import os
import tensorflow as tf
import numpy as np
import pickle
import time
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import neighbors, tree, svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.ops import rnn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import cv2
encoder = OneHotEncoder(sparse=False)

def deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features):
    hidden_1_layer = {'f_fum': hiddenlayersize1,
                      'weight': tf.Variable(tf.random_normal([n_features, hiddenlayersize1])),
                      'bias': tf.Variable(tf.random_normal([hiddenlayersize1]))}
    hidden_2_layer = {'f_fum': hiddenlayersize2,
                      'weight': tf.Variable(tf.random_normal([hiddenlayersize1, hiddenlayersize2])),
                      'bias': tf.Variable(tf.random_normal([hiddenlayersize2]))}
    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([hiddenlayersize2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output

DataSet = np.load('regressiondata/LineMantleCoordinateConvNetData.npy')
IMG_SIZE = 200

dataX = np.asarray([np.ravel(i[0]) for i in DataSet])
datay = np.asarray([i[1] for i in DataSet])

n_features = len(dataX[0])
n_classes = 4
epochs = 10

hiddenlayersize1 = 100
hiddenlayersize2 = 100

train_x, test_x, train_y, test_y = train_test_split(dataX, datay, test_size=0.2)

batch_size = 100

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float')

prediction = deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features)
cost = tf.reduce_mean(tf.squared_difference(prediction, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    estimatedruntime, earlyexitcount, earlyexit = time.time(), 0, 0
    for epoch in range(epochs):
        epoch_loss, i = 0, 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    y_p = tf.argmax(prediction, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x, y: test_y})
    y_pred = y_pred + 1
    trainaccuracy = accuracy.eval({x: train_x, y: train_y})
    testaccuracy = accuracy.eval({x: test_x, y: test_y})
    saver.save(sess, "modelparameters/200x200LineDNNTest.ckpt")
    print(trainaccuracy)
    print(testaccuracy)

sess.close()
tf.reset_default_graph()

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float')
prediction = deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features)

fig = plt.figure()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'modelparameters/200x200LineDNNTest.ckpt')
    for i, imgdata in enumerate(test_x[:10]):
        y = fig.add_subplot(3, 4, i+1)
        img = imgdata.reshape(IMG_SIZE, IMG_SIZE)
        y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [imgdata]}), 1))
        print(y_pred)
        # model_out = np.array([int(model_out[0]*200), int(model_out[1]*200), int(model_out[2]*200), int(model_out[3]*200)])
        # img[model_out[0] - 5:model_out[0] + 5, model_out[1] - 5:model_out[1] + 5] = [0]
        # img[model_out[2] - 5:model_out[2] + 5, model_out[3] - 5:model_out[3] + 5] = [0]
        #
        # if np.argmax(model_out) == 0: str_label = 'Diver'
        # elif np.argmax(model_out) == 1: str_label = 'Fish'
        # elif np.argmax(model_out) == 2: str_label = 'Kelp'
        # else: str_label = 'Octopus'
        #
        # y.imshow(img, cmap='gray')
        # plt.title(str_label)
        # y.axes.get_xaxis().set_visible(False)
        # y.axes.get_yaxis().set_visible(False)
plt.show()



