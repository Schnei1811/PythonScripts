import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import scipy.io.wavfile as wavfile

def recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size):
    layer = {'weights': tf.Variable(tf.random_normal([hiddenlayersize, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, sequence_size])
    x = tf.split(x, n_sequence, 0)
    lstm_cell = rnn.BasicLSTMCell(hiddenlayersize, state_is_tuple=True, forget_bias=0.9)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

def use_RNN(input_data):
    x = tf.placeholder('float', [None, n_sequence, sequence_size])
    prediction = recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model/RNNmodel.ckpt")
        input_data = input_data.reshape((n_sequence, sequence_size))
        y_pred = sess.run(tf.argmax(prediction.eval(feed_dict={x: [input_data]}), 1)) + 1
    sess.close()
    tf.reset_default_graph()
    return y_pred

def Process_Raw_Audio():
    for soundfile in os.listdir(RAW_AUDIO_DIR):
        path = os.path.join(RAW_AUDIO_DIR, soundfile)
        rate, wavdata = wavfile.read(path)
        wavdata = wavdata[:, 0]             #Consider only one channel
        wavdata = wavdata[0::SamplingRate]  #Reduce Data Size
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
        data = data[0:len(data)]
        print(soundfile, use_RNN(data))


RAW_AUDIO_DIR = "RawTestAudio/"
MODEL_DIR = "model/RNNmodel.ckpt"
SamplingRate = np.load('model/MLPModelInfo.npy')[1]

n_features = np.load('model/MLPModelInfo.npy')[0]
n_classes = 5
hiddenlayersize = np.load('model/MLPModelInfo.npy')[2]
n_sequence = 9
sequence_size = 3323

Process_Raw_Audio()
