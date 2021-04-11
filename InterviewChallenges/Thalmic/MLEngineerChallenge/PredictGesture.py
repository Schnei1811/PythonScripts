import tensorflow as tf
import numpy as np
import os
import fnmatch

TotalNumTestExamples = len(fnmatch.filter(os.listdir('TestData'), '*.txt'))
dnnaccuracy = np.loadtxt('modelparameters/DNNmodelaccuracy.txt', delimiter=',')
descriptors = np.loadtxt('modelparameters/nclassesfeatures.txt', delimiter=',')

n_features = int(descriptors[0])
n_classes = int(descriptors[1])

def LoadGestureData(filename):
    InputLine = []
    df = np.loadtxt(filename, delimiter=",")
    for i, j in enumerate(df):
        X = df[i, :]
        InputLine = np.insert(X, 0, InputLine)
    print(InputLine)
    return InputLine

def use_DNN(input_data):
    dnnmodelaccuracy = np.loadtxt("modelparameters/DNNmodelaccuracy.txt", delimiter=",")
    hidden_size = int(dnnmodelaccuracy[3])
    x = tf.placeholder('float', [None, n_features])
    prediction = deep_neural_network_model(x, hidden_size, n_classes, n_features)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "modelparameters/DNNmodel.ckpt")
        result = int((sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1))) + 1)
    sess.close()
    tf.reset_default_graph()
    return result

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


'''
Please place Testing Examples in the TestData folder. There are included examples to help answer any questions.

If you plan on loading invidiual files, modify the file name here
'''

filename = 'TestData/Gesture2Test.txt'
print('Predicted Gesture:', use_DNN(LoadGestureData(filename)))

'''
If you'd like to iteratively load example files from the folder, modify the file name here
'''

for i in range(0, TotalNumTestExamples):
    filename = 'TestData/Gesture{}Test.txt'.format(i+1)
    print('Predicted Gesture:', use_DNN(LoadGestureData(filename)))


'''
If you have any questions, comments, or concerns please let me know
'''