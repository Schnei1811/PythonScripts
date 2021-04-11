import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)


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

DataSet = np.load('FormattedData/fulldata.npy').astype(int)
hiddenlayersize = np.load('model/MLPModelInfo.npy')[2]

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, -1:]

lengthdata = len(X)
n_features = len(X[0])
n_classes = int(max(y) - min(y) + 1)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
epochs = 50

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_y_onehot, test_y_onehot = np.matrix(encoder.fit_transform(train_y)), np.matrix(encoder.fit_transform(test_y))


for i in range(256, 0, -1):
    if len(train_x[:]) % i == 0:
        batch_size = i
        break

for i in range(round((n_features) ** 0.5), 1, -1):
    if (n_features) % i == 0:
        n_sequence = i
        sequence_size = int((n_features) / n_sequence)
        break

print(train_x.shape)
print('Batch Size :', batch_size)
print('Num Sequences :', n_sequence, 'Sequence Size :', sequence_size)

x = tf.placeholder('float', [None, n_sequence, sequence_size])
y = tf.placeholder('float')
prediction = recurrent_neural_network(x, hiddenlayersize, n_sequence, sequence_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = optimizer.minimize(cost)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss, i = 0, 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = np.array(train_x[start:end])
            batch_x = batch_x.reshape((batch_size, n_sequence, sequence_size))
            batch_y = np.array(train_y_onehot[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
        print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    y_p = tf.argmax(prediction, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
    y_pred = y_pred + 1
    trainaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
    testaccuracy = accuracy.eval({x: test_x.reshape((-1, n_sequence, sequence_size)), y: test_y_onehot})
    f1 = metrics.f1_score(test_y, y_pred, average='weighted')
    saver.save(sess, "model/RNNmodel.ckpt")
sess.close()
tf.reset_default_graph()

print('Training Set Accuracy:', trainaccuracy, " (", len(train_y), " samples)")
print('Testing Set Accuracy:', testaccuracy, " (", len(test_y), " samples)")
print('F1 Score:', f1)
print('Confusion Martrix:\n', metrics.confusion_matrix(test_y, y_pred))