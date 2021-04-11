import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
from statistics import mode
from sklearn import metrics
from scipy.special import expit
from tabulate import tabulate

def ExampleCreator(NumGesture, TotalPoorDataCount):
    NumExample = 1
    while True:
        InputLine = []
        try: df = np.loadtxt('TestData/Gesture{}_Example{}.txt'.format(NumGesture, NumExample), delimiter=",")
        except FileNotFoundError: break
        for i, j in enumerate(df):
            X = df[i, :]
            InputLine = np.insert(X, 0, InputLine)
        if NumExample == 1: GestureData = InputLine
        else:
            try: GestureData = np.vstack((GestureData, InputLine))
            except ValueError:
                TotalPoorDataCount += 1
                print('Data Error. Gesture:', NumGesture, ' Example:', NumExample, ' Data dropped')
        NumExample += 1
    return GestureData, TotalPoorDataCount

def GestureDataCreation(TotalPoorDataCount):
    NumExample = 1
    NumGesture = 1
    while True:
        try:
            np.loadtxt('TestData/Gesture{}_Example{}.txt'.format(NumGesture, NumExample), delimiter=",")
            GestureData, TotalPoorDataCount = ExampleCreator(NumGesture, TotalPoorDataCount)
        except FileNotFoundError: break
        Classification = np.ones((len(GestureData),1)) * NumGesture
        GestureData = np.concatenate((GestureData, Classification), axis=1)
        if NumGesture == 1: FinalData = GestureData
        elif NumGesture > 1: FinalData = np.vstack((FinalData, GestureData))
        NumGesture += 1
    return FinalData, TotalPoorDataCount

def use_SNN(input_data):
    snnmodelaccuracy = np.loadtxt("modelparameters/GestureData/SNNmodelaccuracy.txt", delimiter=",")
    hidden_size = int(snnmodelaccuracy[3])
    theta1 = np.matrix(np.reshape(snnclf.x[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
    theta2 = np.matrix(np.reshape(snnclf.x[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(input_data, theta1, theta2)
    result = np.array(np.argmax(h, axis=1)) + 1
    return result

def use_DNN(input_data):
    dnnmodelaccuracy = np.loadtxt("modelparameters/GestureData/DNNmodelaccuracy.txt", delimiter=",")
    hiddenlayersize = int(dnnmodelaccuracy[3])
    x = tf.placeholder('float', [None, n_features])
    prediction = deep_neural_network_model(x, hiddenlayersize, n_classes, n_features)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "modelparameters/GestureData/DNNmodel.ckpt")
        for i,j in enumerate(input_data):
            input_data_line = input_data[i,:]
            y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data_line]}), 1)) + 1
            if i == 0: dnny_pred = y_pred
            elif i > 0: dnny_pred = np.vstack((dnny_pred, y_pred))
    sess.close()
    tf.reset_default_graph()
    return dnny_pred

def use_RNN(input_data):
    rnnmodelaccuracy = np.loadtxt("modelparameters/GestureData/RNNmodelaccuracy.txt", delimiter=",")
    hiddenlayersize, n_sequence, sequence_size = int(rnnmodelaccuracy[3]), int(rnnmodelaccuracy[4]), int(rnnmodelaccuracy[5])
    x = tf.placeholder('float', [None, n_sequence, sequence_size])
    prediction = recurrent_neural_network(x, n_classes, hiddenlayersize, n_sequence, sequence_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "modelparameters/GestureData/RNNmodel.ckpt")
        for i,j in enumerate(input_data):
            input_data_line = input_data[i,:]
            input_data_line = input_data_line.reshape((n_sequence, sequence_size))
            y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data_line]}), 1)) + 1
            if i == 0: rnny_pred = y_pred
            elif i > 0: rnny_pred = np.vstack((rnny_pred, y_pred))
    sess.close()
    tf.reset_default_graph()
    return rnny_pred

def use_BiNN(input_data):
    binnmodelaccuracy = np.loadtxt("modelparameters/GestureData/BiNNmodelaccuracy.txt", delimiter=",")
    hiddenlayersize, n_sequence, sequence_size = int(binnmodelaccuracy[3]), int(binnmodelaccuracy[4]), int(binnmodelaccuracy[5])
    x = tf.placeholder('float', [None, n_sequence, n_sequence])
    prediction = bidirectional_recurrent_neural_network(x, n_sequence, sequence_size, hiddenlayersize)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "modelparameters/GestureData/BiNNmodel.ckpt")
        for i,j in enumerate(input_data):
            input_data_line = input_data[i,:]
            input_data_line = input_data_line.reshape((n_sequence, n_sequence))
            y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data_line]}), 1)) + 1
            if i == 0: binny_pred = y_pred
            elif i > 0: binny_pred = np.vstack((binny_pred, y_pred))
    sess.close()
    tf.reset_default_graph()
    return binny_pred

def sigmoid(z):
    return expit(z)

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def deep_neural_network_model(data, hidden_size, n_classes, n_features):
    n_nodes_hl1 = hidden_size
    n_nodes_hl2 = hidden_size
    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output

def recurrent_neural_network(x, n_classes, hiddenlayersize, n_sequence, sequence_size):
    layer = {'weights': tf.Variable(tf.random_normal([hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

def bidirectional_recurrent_neural_network(x, sequence_size, n_sequence, hiddenlayersize):
    layer = {'weights': tf.Variable(tf.random_normal([2*hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_fw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 1.0)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 1.0)
    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], layer['weights']) + layer['biases']

def use_Ensemble(knnclf, logitclf, snny_pred, dnny_pred, rnny_pred):
    snny_pred = [l[0] for l in snny_pred]
    dnny_pred = [l[0] for l in dnny_pred]
    rnny_pred = [l[0] for l in rnny_pred]
    confidence = []
    ensemble = np.vstack((knnclf.predict(X), logitclf.predict(X), snny_pred, dnny_pred, rnny_pred)).T
    for i,j in enumerate(ensemble):
        try: y_pred = mode(ensemble[i,:])
        except: y_pred = ensemble[i,3]
        if i == 0: ensembley_pred = y_pred
        if i > 0: ensembley_pred = np.vstack((ensembley_pred, y_pred))
    return ensembley_pred, ensemble, confidence

TotalPoorDataCount = 0
DataSet, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount)
if TotalPoorDataCount > 0:
    print('Total examples dropped due to less than 1xdif-50 lines of Data:', TotalPoorDataCount)

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, cols - 1:cols]

n_features = len(X[0])
n_classes = int(max(y))

knnclf = pd.read_pickle('modelparameters/GestureData/KNNpickle.pickle')
mnbclf = pd.read_pickle('modelparameters/GestureData/MNBpickle.pickle')
logitclf = pd.read_pickle('modelparameters/GestureData/LOGITpickle.pickle')
snnclf = pd.read_pickle('modelparameters/GestureData/SNNpickle.pickle')

knnaccuracy, knnf1 = knnclf.score(X, y), metrics.f1_score(y, knnclf.predict(X), average='weighted')
mnbaccuracy, mnbf1 = mnbclf.score(X, y), metrics.f1_score(y, mnbclf.predict(X), average='weighted')
logitaccuracy, logitf1 = logitclf.score(X, y), metrics.f1_score(y, logitclf.predict(X), average='weighted')

snny_pred = use_SNN(X)
testcorrect = [1 if a == b else 0 for (a, b) in zip(snny_pred, y)]
snnaccuracy, snnf1 = (sum(map(int, testcorrect)) / len(testcorrect)), metrics.f1_score(y, snny_pred, average='weighted')

dnny_pred = use_DNN(X)
testcorrect = [1 if a == b else 0 for (a, b) in zip(dnny_pred, y)]
dnnaccuracy, dnnf1 = (sum(map(int, testcorrect)) / len(testcorrect)), metrics.f1_score(y, dnny_pred, average='weighted')

rnny_pred = use_RNN(X)
testcorrect = [1 if a == b else 0 for (a, b) in zip(rnny_pred, y)]
rnnaccuracy, rnnf1 = (sum(map(int, testcorrect)) / len(testcorrect)), metrics.f1_score(y, rnny_pred, average='weighted')

binny_pred = use_BiNN(X)
testcorrect = [1 if a == b else 0 for (a, b) in zip(binny_pred, y)]
binnaccuracy, binnf1 = (sum(map(int, testcorrect)) / len(testcorrect)), metrics.f1_score(y, binny_pred, average='weighted')

ensembley_pred, ensemble, confidence = use_Ensemble(knnclf, logitclf, snny_pred, dnny_pred, rnny_pred)
testcorrect = [1 if a == b else 0 for (a, b) in zip(ensembley_pred, y)]
ensaccuracy, ensf1 = (sum(map(int, testcorrect)) / len(testcorrect)), metrics.f1_score(y, ensembley_pred, average='weighted')

print('\nTotal Number of Testing Examples: ', len(X))

print('\nModel Test Data Accuracy & F1 Score:\n')

print(tabulate([['K-Nearest Neighbour', round(knnaccuracy, 3), round(knnf1,3)],
                ['Multinomial Naive Bayes', round(mnbaccuracy, 3), round(mnbf1,3)],
                ['Logistic Regression', round(logitaccuracy, 3), round(logitf1,3)],
                ['Simple Neural Network', round(snnaccuracy, 3), round(snnf1,3)],
                ['Deep Neural Network', round(dnnaccuracy, 3), round(dnnf1,3)],
                ['Recurrent Neural Network', round(rnnaccuracy, 3), round(rnnf1,3)],
                ['BiDirectional Neural Network', round(binnaccuracy, 3), round(binnf1,3)],
                ['Ensemble Model', round(knnaccuracy, 3), round(knnf1,3)]],
                headers=['Model Name', 'Accuracy', 'F1 Score']))

print('\n\nEnsemble Method uses the mode of:\n\n\tK-Nearest Neighbour, Logistic Regression, Simple Neural Network, \n\tDeep Neural Network, '
      'and Recurrent Neural Network.\n\nIn the case of a tie the Deep Neural Network result is submitted.')

ans = input('\nWould you like to see detailed results of the five model ensemble? (y/n) ')
ans = 'y'
if ans in ['y']:
    print('\nThe following examples were misclassified by the ensemble method.\nThe number of * indicate the number of misclassifications.\n'
          'Consider reviewing example Data with **** or *****\n')
    examplenum = 1
    for i,j in enumerate(ensemble):
        if i > 0:
            if y[i] - y[i-1] != 0: examplenum = 1
        if (list(ensemble[i,:5]).count(y[i])) == 2: print('Gesture', int(y[i]), ' Example', examplenum, ' \t***')
        elif (list(ensemble[i,:5]).count(y[i])) == 1: print('Gesture', int(y[i]), ' Example', examplenum, ' \t****')
        elif (list(ensemble[i, :5]).count(y[i])) == 0: print('Gesture', int(y[i]), ' Example', examplenum, ' \t*****')
        examplenum += 1