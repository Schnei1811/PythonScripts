import numpy as np
import tensorflow as tf
import os
import scipy.io.wavfile as wavfile

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

def use_DNN(input_data):
    x = tf.placeholder('float', [None, n_features])
    prediction = deep_neural_network_model(x, hiddenlayersize, n_classes, n_features)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_DIR)
        y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)) + 1
    sess.close()
    tf.reset_default_graph()
    return y_pred

def Process_Raw_Audio():
    for soundfile in os.listdir(RAW_AUDIO_DIR):
        path = os.path.join(RAW_AUDIO_DIR, soundfile)
        rate, wavdata = wavfile.read(path)
        i, audiostart, audioend, audiostarttoggle = 0, 0, 0, False
        for sound in wavdata:
            if -500 < sound < 500: pass
            else:
                if audiostarttoggle == False:
                    audiostarttoggle = True
                    audiostart = i
                audioend = i
            i += 1
        outputdata = wavdata[audiostart:audioend]
        data = np.lib.pad(outputdata, (0, n_features - len(outputdata)), 'constant', constant_values=(0, 0))
        print(soundfile, use_DNN(data))


RAW_AUDIO_DIR = "RawTestAudio/"
MODEL_DIR = "model/DNNmodel.ckpt"


n_features = np.load('model/modelinfo.npy')[0]
n_classes = 5
hiddenlayersize = 100


Process_Raw_Audio()
