import os
import tensorflow as tf
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn import neighbors, tree, svm, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow.python.ops import rnn, rnn_cell
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
snnepochcount = 0

if not os.path.exists("./Data"):
    print("\nA Data folder has been created in this directory.\nPlease place a supervised learning dataset in this folder. All outputs must be in the far right columns.\n")
    os.makedirs("./Data")

def TrainKNearestNeighbor(train_x, test_x, train_y, test_y):
    starttime = time.time()
    knnclf = neighbors.KNeighborsClassifier(n_jobs=-1)
    knnclf.fit(train_x, train_y)
    trainaccuracy = knnclf.score(train_x, train_y)
    testaccuracy = knnclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, knnclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'KNN'
    ans = Save(knnclf, trainaccuracy, testaccuracy, f1, modelname, knnprevaccuracy, knnprevf1score, runtimeseconds)
    if ans in ['y']:
        modelparam = np.array([testaccuracy, f1, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')

def TrainRandomForest(train_x, test_x, train_y, test_y):
    starttime = time.time()
    rfclf = tree.DecisionTreeClassifier()
    rfclf.fit(train_x, train_y)
    trainaccuracy = rfclf.score(train_x, train_y)
    testaccuracy = rfclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, rfclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'RF'
    ans = Save(rfclf, trainaccuracy, testaccuracy, f1, modelname, rfprevaccuracy, rfprevf1score, runtimeseconds)
    if ans in ['y']:
        modelparam = np.array([testaccuracy, f1, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')

def TrainMultinomialNaiveBayes(train_x, test_x, train_y, test_y):
    starttime = time.time()
    mnbclf = MultinomialNB()
    try: mnbclf.fit(train_x, train_y)
    except ValueError:
        print('Multinomial Naive Bayes does not work with negative feature values')
        ModelSelect()
        exit()
    trainaccuracy = mnbclf.score(test_x, test_y)
    testaccuracy = mnbclf.score(train_x, train_y)
    f1 = metrics.f1_score(test_y, mnbclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'MNB'
    ans = Save(mnbclf,trainaccuracy,testaccuracy, f1, modelname, mnbprevaccuracy, mnbprevf1score, runtimeseconds)
    if ans in ['y']:
        modelparam = np.array([testaccuracy, f1, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')

def TrainSVM(train_x, test_x, train_y, test_y, C):
    starttime = time.time()
    svmclf = svm.SVC(C=C, kernel='rbf', max_iter = -1)
    svmclf.fit(train_x, train_y)
    trainaccuracy = svmclf.score(train_x, train_y)
    testaccuracy = svmclf.score(test_x, test_y)
    f1 = metrics.f1_score(test_y, svmclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'SVM'
    ans = Save(svmclf, trainaccuracy, testaccuracy, f1, modelname, svmprevaccuracy, svmprevf1score, runtimeseconds)
    if ans in ['y']:
        modelparam = np.array([testaccuracy, f1, C, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')

def TrainLogisticRegression(train_x, test_x, train_y, test_y, C):
    starttime = time.time()
    logitclf = LogisticRegression(C = C)
    logitclf.fit(train_x, train_y)
    trainaccuracy = logitclf.score(test_x, test_y)
    testaccuracy = logitclf.score(train_x, train_y)
    f1 = metrics.f1_score(test_y, logitclf.predict(test_x), average='weighted')
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'LOGIT'
    ans = Save(logitclf, trainaccuracy, testaccuracy, f1, modelname, logitprevaccuracy, logitprevf1score, runtimeseconds)
    if ans in ['y']:
        modelparam = np.array([testaccuracy, f1, C, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')

def TrainSimpleNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hidden_size, regularization):
    global snnepochcount
    starttime = time.time()
    train_x, test_x = np.matrix(train_x), np.matrix(test_x)
    train_y_onehot = np.matrix(encoder.fit_transform(train_y))

    params = (np.random.random(size=hidden_size * (n_features + 1) + n_classes * (hidden_size + 1)) - 0.5) * 0.25
    fmin = minimize(fun=backprop, x0=params, args=(n_features, hidden_size, n_classes, train_x, train_y_onehot, regularization, epochs),
                    method='TNC', jac=True, options={'maxiter': epochs})

    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(train_x, theta1, theta2)
    y_predtrain = np.array(np.argmax(h, axis=1) + 1)
    traincorrect = [1 if a == b else 0 for (a, b) in zip(y_predtrain, train_y)]
    trainaccuracy = (sum(map(int, traincorrect)) / len(traincorrect))

    a1, z2, a2, z3, h = forward_propagate(test_x, theta1, theta2)
    y_predtest = np.array(np.argmax(h, axis=1) + 1)
    testcorrect = [1 if a == b else 0 for (a, b) in zip(y_predtest, test_y)]
    testaccuracy = (sum(map(int, testcorrect)) / len(testcorrect))

    f1 = metrics.f1_score(test_y, y_predtest, average='weighted')
    snnepochcount = 0
    runtimeseconds = round((time.time() - starttime), 2)
    modelname = 'SNN'
    ans = Save(y_predtest, trainaccuracy, testaccuracy, f1, modelname, snnprevaccuracy, snnprevf1score, runtimeseconds)
    if ans in ['y']:
        savepickle = open('modelparameters/{}/SNNpickle.pickle'.format(dataname), 'wb')
        pickle.dump(fmin, savepickle)
        modelparam = np.array([testaccuracy, f1, epochs, hidden_size, regularization, runtimeseconds])
        np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')
        print("Model Saved")
    else:
        print("Model Unsaved")

def TrainDeepNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize):
    starttime = time.time()
    batch_size = 100

    train_y_onehot = np.matrix(encoder.fit_transform(train_y))
    test_y_onehot = np.matrix(encoder.fit_transform(test_y))

    x = tf.placeholder('float', [None, n_features])
    y = tf.placeholder('float')

    prediction = deep_neural_network_model(x, hiddenlayersize, n_classes, n_features)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        estimatedruntime = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y_onehot[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x, y: test_y_onehot})
        y_pred = y_pred+1
        trainaccuracy = accuracy.eval({x: train_x, y: train_y_onehot})
        testaccuracy = accuracy.eval({x: test_x, y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'DNN'
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, dnnprevaccuracy, dnnprevf1score, runtimeseconds)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/DNNmodel.ckpt".format(dataname))
            modelparam = np.array([testaccuracy, f1, epochs, hiddenlayersize, runtimeseconds])
            np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')
            print("Model Saved")
        else: print("Model Unsaved")
    sess.close()
    tf.reset_default_graph()

def TrainRecurrentNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize):
    starttime = time.time()
    train_y_onehot = np.matrix(encoder.fit_transform(train_y))
    test_y_onehot = np.matrix(encoder.fit_transform(test_y))

    for i in range(256, 0, -1):
        if len(train_x[:]) % i == 0:
            batch_size = i
            break

    for i in range(round((n_features) ** 0.5), 1, -1):
        if (n_features) % i == 0:
            sequence_size = i
            n_sequence = int((n_features) / sequence_size)
            break

    print("Batch Size: ", batch_size, "\nNumber of Sequences: ", n_sequence, "\nSequence Size: ", sequence_size)

    x = tf.placeholder('float', [None, n_sequence, sequence_size])
    y = tf.placeholder('float')

    prediction = recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        estimatedruntime = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size, n_sequence, sequence_size))
                batch_y = np.array(train_y_onehot[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p],feed_dict={x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        y_pred = y_pred+1
        trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'RNN'
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, rnnprevaccuracy, rnnprevf1score, runtimeseconds)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/RNNmodel.ckpt".format(dataname))
            modelparam = np.array([testaccuracy, f1, epochs, hiddenlayersize, n_sequence, sequence_size, runtimeseconds])
            np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')
            print("Model Saved")
        else: print("Model Unsaved")
    sess.close()
    tf.reset_default_graph()

def TrainBiDirectionalRecurrentNeuralNetwork(train_x, test_x, train_y, test_y, epochs, hiddenlayersize):
    starttime = time.time()
    train_y_onehot = np.matrix(encoder.fit_transform(train_y))
    test_y_onehot = np.matrix(encoder.fit_transform(test_y))

    for i in range(256, 0, -1):
        if len(train_x[:]) % i == 0:
            batch_size = i
            break

    for i in range(round((n_features) ** 0.5), 1, -1):
        if (n_features) % i == 0:
            sequence_size = i
            n_sequence = int((n_features) / sequence_size)
            break

    print("Batch Size: ", batch_size, "\nNumber of Sequences: ", n_sequence, "\nSequence Size: ", sequence_size)

    x = tf.placeholder("float", [None, n_sequence, sequence_size])
    y = tf.placeholder("float", [None, n_classes])

    prediction = BiRNN(x, sequence_size, n_sequence, hiddenlayersize)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        estimatedruntime = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_x = batch_x.reshape((batch_size, sequence_size, n_sequence))
                batch_y = np.array(train_y_onehot[start:end])
                batch_x = batch_x.reshape((batch_size, n_sequence, sequence_size))
                _, c = sess.run([optimizer,cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            if epoch == 0:
                estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
                print('\nEstimated Model Running Time:', round(estimatedruntime, 2), timemetric, '\n')
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        y_p = tf.argmax(prediction, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p],feed_dict={x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        y_pred = y_pred + 1
        trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
        f1 = metrics.f1_score(test_y, y_pred, average='weighted')
        runtimeseconds = round((time.time() - starttime), 2)
        modelname = 'BiNN'
        ans = Save(y_pred, trainaccuracy, testaccuracy, f1, modelname, binnprevaccuracy, binnprevf1score, runtimeseconds)
        if ans in ["y"]:
            saver.save(sess, "modelparameters/{}/BiNNmodel.ckpt".format(dataname))
            modelparam = np.array([testaccuracy, f1, epochs, hiddenlayersize, n_sequence, sequence_size, runtimeseconds])
            np.savetxt('modelparameters/{}/{}modelaccuracy.txt'.format(dataname, modelname), modelparam[None], fmt='%.5f', delimiter=',')
            print("Model Saved")
        else: print("Model Unsaved")
    sess.close()
    tf.reset_default_graph()

def sigmoid(z):
    return expit(z)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def backprop(params, n_features, hidden_size, n_classes, X, y, learning_rate, epochs):
    global snnepochcount
    estimatedruntime = time.time()
    m = X.shape[0]
    J = 0
    X, y = np.matrix(X), np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (n_features + 1)], (hidden_size, (n_features + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (n_features + 1):], (n_classes, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    delta1, delta2 = np.zeros(theta1.shape), np.zeros(theta2.shape)
    for i in range(m):
        try: first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        except ValueError: print("Check if you have classifications that are labelled 0. If so, +1 to Classifications and rerun")
        try: second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        except ValueError: print("Check if you have input Data without a labelled classifications")
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    for t in range(m):
        a1t = a1[t, :]
        z2t = z2[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = y[t, :]
        d3t = ht - yt
        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
    delta1, delta2 = delta1 / m, delta2 / m
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    if snnepochcount == 0:
        estimatedruntime, timemetric = TimeConversion((time.time() - estimatedruntime) * epochs)
        print('\nEstimated Model Running Time:', round(estimatedruntime,2), timemetric, '\n')
    print("Iteration:", snnepochcount, " Cost:", J)
    snnepochcount = snnepochcount + 1
    return J, grad

def deep_neural_network_model(x, hiddenlayersize, n_classes, n_features):
    n_nodes_hl1 = hiddenlayersize
    n_nodes_hl2 = hiddenlayersize
    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }
    l1 = tf.add(tf.matmul(x, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output

# def recurrent_neural_network(x, hiddenlayersize, n_chunks, chunk_size):
#     n_nodes_hl1 = hiddenlayersize
#     n_nodes_hl2 = hiddenlayersize
#          layer = {'weights': tf.Variable(tf.random_normal([hiddenlayersize, n_classes])),
#                   'biases': tf.Variable(tf.random_normal([n_classes]))}
#     x = tf.transpose(x, [1, 0, 2])
#     x = tf.reshape(x, [-1, chunk_size])
#     x = tf.split(0, n_chunks, x)
#     lstm_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True)
#     stacked_lstm = rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)
#     outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
#     hidden_1_layer = {'f_fum': n_nodes_hl1,
#                       'weight': tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
#                       'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}
#     hidden_2_layer = {'f_fum': n_nodes_hl2,
#                       'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                       'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
#     output_layer = {'f_fum': None,
#                     'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
#                     'bias': tf.Variable(tf.random_normal([n_classes])), }
#     l1 = tf.add(tf.matmul(x, hidden_1_layer['weight']), hidden_1_layer['bias'])
#     l1 = tf.nn.relu(l1)
#     l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
#     l2 = tf.nn.relu(l2)
#     output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
#     return output

def recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size):
    layer = {'weights': tf.Variable(tf.random_normal([hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

def BiRNN(x, sequence_size, n_sequence, hiddenlayersize):
    layer = {'weights': tf.Variable(tf.random_normal([2*hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(0, n_sequence, x)
    lstm_fw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 1.0)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias = 1.0)
    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], layer['weights']) + layer['biases']

def isPrime(n):  # Return True if Prime
    if n == 2: return True
    if n == 3: return True
    if n % 2 == 0: return False
    if n % 3 == 0: return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0: return False
        i += w
        w = 6 - w
    return True

def Save(clf, trainaccuracy, testaccuracy, f1score, modelname, prevaccuracy, prevf1score, runtimeseconds):
    runtime, timemetric = TimeConversion(runtimeseconds)
    print('\n\nTime: {}{}'.format(round(runtime,2), timemetric))
    print('Testing Set Accuracy:', testaccuracy)
    print('Accuracy Change from Saved Model:', testaccuracy - prevaccuracy)
    print('\nF1 Score:', f1score)
    print('F1 Score Change from Saved Model:', f1score - prevf1score)
    if (testaccuracy - prevaccuracy) > 0 and (f1score - prevf1score) > 0:
        print('\nYour results have improved!')
    print('\nTraining Set Accuracy:', trainaccuracy)
    print('Training Set Accuracy - Testing Set Accuracy:', trainaccuracy - testaccuracy)
    if modelname in ['SVM', 'SNN', 'DNN', 'RNN']:
        if trainaccuracy < 0.9:
            print('\nModel suffering from high bias. Consider decreasing regularization, a polynomial expansion dataset or getting more features')
        if trainaccuracy >= 0.9 and testaccuracy < 0.9:
            print('\nModel suffering from high variance. Consider increasing regularization, getting more Data, or reducing features')
    if modelname in ['KNN','RF','MNB','SVM','LOGIT']:
        print('\nConfusion Matrix:\n', metrics.confusion_matrix(test_y, clf.predict(test_x)))
        ans = input("\nWould you like to save the new parameters? (y/n): ")
        if ans in ["y"]:
            savepickle = open('modelparameters/{}/{}pickle.pickle'.format(dataname, modelname), 'wb')
            pickle.dump(clf, savepickle)
            print("Model Saved")
        elif ans in ['n']: print("Model Unsaved")
        else:
            print("Improper Command")
            Save(clf, trainaccuracy, testaccuracy, f1score, modelname, prevaccuracy, prevf1score, runtimeseconds)
    elif modelname in['SNN', 'DNN', 'RNN', 'BiNN']:
        print('\nConfusion Martrix:\n', metrics.confusion_matrix(test_y, clf))
        ans = input("\nWould you like to save the new parameters? (y/n): ")
    return ans

def TimeConversion(runtimeseconds):
    timemetric = "s"
    if runtimeseconds >= 60 and runtimeseconds < 3600:
        totaltimer = runtimeseconds / 60
        timemetric = "m"
    elif runtimeseconds >= 3600:
        totaltimer = runtimeseconds / 3600
        timemetric = "h"
    else: totaltimer = runtimeseconds
    return totaltimer, timemetric

def ModelSelect():
    ans = input('\nTrain which model? (knn, rf, mnb, svm, logit, snn, dnn, rnn, binn, or help, exit):')
    if ans in ['knn']:
        TrainKNearestNeighbor(train_x, test_x, train_y, test_y)
    elif ans in ['rf']:
        TrainRandomForest(train_x, test_x, train_y, test_y)
    elif ans in ['mnb']:
        TrainMultinomialNaiveBayes(train_x, test_x, train_y, test_y)
    elif ans in ['svm']:
        C = float(input('Enter Value for C (1 default):'))
        TrainSVM(train_x, test_x, train_y, test_y, C)
    elif ans in ['logit']:
        C = float(input('Enter Value for C (1 default):'))
        TrainLogisticRegression(train_x, test_x, train_y, test_y, C)
    elif ans in ['snn']:
        epochs = int(input('How many epochs? '))
        hiddenlayersize = int(input('Hidden Layer Size? (Recommended ' + str(2 * n_features) + ') '))
        regularization = float(input('Enter Value for Regularization (1 default) '))
        TrainSimpleNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize, regularization)
    elif ans in ['dnn']:
        epochs = int(input('How many epochs? '))
        hiddenlayersize = int(input('Hidden Layer Size? (Recommended ' + str(2 * n_features) + ') '))
        TrainDeepNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize)
    elif ans in ['rnn']:
        epochs = int(input('How many epochs? '))
        rnn_cellsize = int(input('Hidden Layer Size? (Recommended ' + str(2 * n_features) + ') '))
        TrainRecurrentNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, rnn_cellsize)
    elif ans in ['binn']:
        epochs = int(input('How many epochs? '))
        hiddenlayersize = int(input('Hidden Layer Size? (Recommended ' + str(2 * n_features) + ') '))
        TrainBiDirectionalRecurrentNeuralNetwork(train_x, test_x, nntrain_y, test_y, epochs, hiddenlayersize)
    elif ans in ['help']:
        print('\nModel Descriptions:\nknn: K-Nearest Neighbour\nrf: Random Forest\nmnb: Multinomial Naive Bayes\nsvm: Radial Basis Function Support Vector Machine\n'
              'logit: Logistic Regression\nsnn: Simple Neural Network\ndnn: Deep Neural Network (Multilayer Percepetron)\nrnn: Long Short-Term Memory Recurrent Neural Network'
              '\nbinn BiDirectional Long-Short Term Memory Recurrent Neural Network')
    elif ans in ['exit']: exit()
    else:
        print('Improper Selection')
        ModelSelect()

def LoadData():
    dirs = os.listdir("Data/")
    print("\nAvailable Datasets: \n")
    for file in dirs:
        print(file)
    datarun = input("\nEnter a Dataset:")
    return datarun


datarun = LoadData()
dataname = datarun[:-4]
try: DataSet = np.loadtxt('Data/{}'.format(datarun), delimiter=",")
except FileNotFoundError:
    try:
        DataSet = np.loadtxt('Data/{}.txt'.format(datarun), delimiter=",")
        dataname = datarun
    except FileNotFoundError:
        try:
            DataSet = np.loadtxt('Data/{}.csv'.format(datarun), delimiter=",")
            dataname = datarun
        except FileNotFoundError:
            print('Dataset Not Found')
            LoadData()

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, cols - 1:cols]

lengthdata = len(X)
n_features = len(X[0])
n_classes = int(max(y) - min(y) + 1)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
nntrain_y = train_y
train_y = np.ravel(train_y)

if not os.path.exists("./modelparameters/{}".format(dataname)):
    os.makedirs("./modelparameters/{}".format(dataname))
file = open("modelparameters/{}/nclassesfeatures.txt".format(dataname), "w")
file.write(str(n_classes) + "," + str(n_features) + "," + str(lengthdata))
file.flush()

try:
    knnmodelaccuracy = np.loadtxt("modelparameters/{}/KNNmodelaccuracy.txt".format(dataname), delimiter=",")
    knnprevaccuracy, knnprevf1score, knntime = round(knnmodelaccuracy[0], 5), round(knnmodelaccuracy[1], 5), round(knnmodelaccuracy[2], 5)
    knntime, knntimemetric = TimeConversion(knntime)
except FileNotFoundError: knnprevaccuracy, knnprevf1score, knntime, knntimemetric = 0, 0, 0, ''
try:
    rfmodelaccuracy = np.loadtxt("modelparameters/{}/RFmodelaccuracy.txt".format(dataname), delimiter=",")
    rfprevaccuracy, rfprevf1score, rftime = round(rfmodelaccuracy[0], 5), round(rfmodelaccuracy[1], 5), round(rfmodelaccuracy[2], 5)
    rftime, rftimemetric = TimeConversion(rftime)
except FileNotFoundError: rfprevaccuracy, rfprevf1score, rftime, rftimemetric = 0, 0, 0, ''
try:
    mnbmodelaccuracy = np.loadtxt("modelparameters/{}/MNBmodelaccuracy.txt".format(dataname), delimiter=",")
    mnbprevaccuracy, mnbprevf1score, mnbtime = round(mnbmodelaccuracy[0], 5), round(mnbmodelaccuracy[1], 5), round(mnbmodelaccuracy[2], 5)
    mnbtime, mnbtimemetric = TimeConversion(mnbtime)
except FileNotFoundError: mnbprevaccuracy, mnbprevf1score, mnbtime, mnbtimemetric = 0, 0, 0, ''
try:
    svmmodelaccuracy = np.loadtxt("modelparameters/{}/SVMmodelaccuracy.txt".format(dataname), delimiter=",")
    svmprevaccuracy, svmprevf1score, svmC, svmtime = round(svmmodelaccuracy[0], 5), round(svmmodelaccuracy[1], 5), round(svmmodelaccuracy[2], 5), round(svmmodelaccuracy[3], 5)
    svmtime, svmtimemetric = TimeConversion(svmtime)
except FileNotFoundError: svmprevaccuracy, svmprevf1score, svmC, svmtime, svmtimemetric = 0, 0, 0, 0, ''
try:
    logitmodelaccuracy = np.loadtxt("modelparameters/{}/LOGITmodelaccuracy.txt".format(dataname), delimiter=",")
    logitprevaccuracy, logitprevf1score, logitC, logittime = round(logitmodelaccuracy[0], 5), round(logitmodelaccuracy[1], 5), round(logitmodelaccuracy[2], 5), round(logitmodelaccuracy[3], 5)
    logittime, logittimemetric = TimeConversion(logittime)
except FileNotFoundError: logitprevaccuracy, logitprevf1score, logitC, logittime, logittimemetric = 0, 0, 0, 0, ''
try:
    snnmodelaccuracy = np.loadtxt("modelparameters/{}/SNNmodelaccuracy.txt".format(dataname), delimiter=",")
    snnprevaccuracy, snnprevf1score, snnepochs, snnhiddenlayersize, snnregularization, snntime = round(snnmodelaccuracy[0], 5), round(snnmodelaccuracy[1], 5), int(snnmodelaccuracy[2]), int(snnmodelaccuracy[3]), round(snnmodelaccuracy[4], 5), round(snnmodelaccuracy[5], 5)
    snntime, snntimemetric = TimeConversion(snntime)
except FileNotFoundError: snnprevaccuracy, snnprevf1score, snnepochs, snnhiddenlayersize, snnregularization, snntime, snntimemetric = 0, 0, 0, 0, 0, 0, ''
try:
    dnnmodelaccuracy = np.loadtxt("modelparameters/{}/DNNmodelaccuracy.txt".format(dataname), delimiter=",")
    dnnprevaccuracy, dnnprevf1score, dnnepochs, dnnhiddenlayersize, dnntime = round(dnnmodelaccuracy[0], 5), round(dnnmodelaccuracy[1], 5), int(dnnmodelaccuracy[2]), int(dnnmodelaccuracy[3]), round(dnnmodelaccuracy[4], 5)
    dnntime, dnntimemetric = TimeConversion(dnntime)
except FileNotFoundError: dnnprevaccuracy, dnnprevf1score, dnnepochs, dnnhiddenlayersize, dnntime, dnntimemetric = 0, 0, 0, 0, 0, ''
try:
    rnnmodelaccuracy = np.loadtxt("modelparameters/{}/RNNmodelaccuracy.txt".format(dataname), delimiter=",")
    rnnprevaccuracy, rnnprevf1score, rnnepochs, rnnhiddenlayersize, rnnnsequence, rnnsequencesize, rnntime  = round(rnnmodelaccuracy[0], 5), round(rnnmodelaccuracy[1], 5), int(rnnmodelaccuracy[2]), int(rnnmodelaccuracy[3]), int(rnnmodelaccuracy[4]), int(rnnmodelaccuracy[5]), round(rnnmodelaccuracy[6], 5)
    rnntime, rnntimemetric = TimeConversion(rnntime)
except FileNotFoundError: rnnprevaccuracy, rnnprevf1score, rnnepochs, rnnhiddenlayersize,rnnnsequence, rnnsequencesize,  rnntime, rnntimemetric = 0, 0, 0, 0, 0, 0, 0, ''
try:
    binnmodelaccuracy = np.loadtxt("modelparameters/{}/BiNNmodelaccuracy.txt".format(dataname), delimiter=",")
    binnprevaccuracy, binnprevf1score, binnepochs, binnhiddenlayersize, binnnsequence, binnsequencesize, binntime  = round(binnmodelaccuracy[0], 5), round(binnmodelaccuracy[1], 5), int(binnmodelaccuracy[2]), int(binnmodelaccuracy[3]), int(binnmodelaccuracy[4]), int(binnmodelaccuracy[5]), round(binnmodelaccuracy[6], 5)
    binntime, binntimemetric = TimeConversion(binntime)
except FileNotFoundError: binnprevaccuracy, binnprevf1score, binnepochs, binnhiddenlayersize, binnnsequence, binnsequencesize, binntime, binntimemetric = 0, 0, 0, 0, 0, 0, 0, ''

print('\nData Length:', lengthdata)
print('Number of Features:', n_features)
print('Number of Classes:', n_classes)
print("\nCurrent Scores:")
print("K-Nearest Neighbour: \tAccuracy", knnprevaccuracy, "\tF1", knnprevf1score, "\tTime", round(knntime,2), knntimemetric,)
print("Random Forest: \t\t\tAccuracy", rfprevaccuracy, "\tF1", rfprevf1score, "\tTime", round(rftime,2), rftimemetric)
print("Multnomial Naive Bayes: Accuracy", mnbprevaccuracy, "\tF1", mnbprevf1score, "\tTime", round(mnbtime,2), mnbtimemetric)
print("RBF SVM: \t\t\t\tAccuracy", svmprevaccuracy, "\tF1", svmprevf1score, '\tC:', svmC, "\t\t\tTime", round(svmtime,2), svmtimemetric)
print("Logistic Regression: \tAccuracy", logitprevaccuracy, "\tF1", logitprevf1score, '\tC:', logitC, "\t\t\tTime", round(logittime,2), logittimemetric)
print("Simple Neural Network: \tAccuracy", snnprevaccuracy, "\tF1", snnprevf1score, '\tEpochs', snnepochs, '\t\tHLSize', snnhiddenlayersize, '\t\tReg', snnregularization, "\t\tTime", round(snntime,2), snntimemetric)
print("Deep NN: \t\t\t\tAccuracy", dnnprevaccuracy, "\tF1", dnnprevf1score, "\tEpochs", dnnepochs, "\tHLSize", dnnhiddenlayersize, "\tTime", round(dnntime,2), dnntimemetric)
print("LSTM RNN: \t\t\t\tAccuracy", rnnprevaccuracy, "\tF1", rnnprevf1score, "\tEpochs", rnnepochs, "\tHLSize", rnnhiddenlayersize, "\tNum Sequences", rnnnsequence, "\tSequence Size", rnnsequencesize,  "\tTime", round(rnntime,2), rnntimemetric)
print("BiDirectional RNN: \t\tAccuracy", binnprevaccuracy, "\tF1", binnprevf1score, "\tEpochs", binnepochs, "\tHLSize", binnhiddenlayersize,"\tNum Sequences", binnnsequence, "\tSequence Size", binnsequencesize,  "\tTime", round(binntime,2), binntimemetric)


ModelSelect()

while True:
    ans = input("\nTrain Data with Alternative Model? (y/n): ")
    if ans in ["y"]: ModelSelect()
    else: exit()