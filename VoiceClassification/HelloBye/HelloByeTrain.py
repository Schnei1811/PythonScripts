import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
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

DataSet = np.load('FormattedData/fulldata.npy').astype(int)

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, -1:]

lengthdata = len(X)
n_features = len(X[0])
n_classes = int(max(y) - min(y) + 1)

hiddenlayersize1 = 100
hiddenlayersize2 = 100
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
epochs = 10

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

train_y_onehot, test_y_onehot = np.matrix(encoder.fit_transform(train_y)), np.matrix(encoder.fit_transform(test_y))
batch_size = 100

print(train_x.shape)

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float')

prediction = deep_neural_network_model(x, hiddenlayersize1, hiddenlayersize2, n_classes, n_features)
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
            batch_y = np.array(train_y_onehot[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
        print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    y_p = tf.argmax(prediction, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x, y: test_y_onehot})
    y_pred = y_pred + 1
    trainaccuracy = accuracy.eval({x: train_x, y: train_y_onehot})
    testaccuracy = accuracy.eval({x: test_x, y: test_y_onehot})
    f1 = metrics.f1_score(test_y, y_pred, average='weighted')
    saver.save(sess, "model/DNNmodel.ckpt")
sess.close()
tf.reset_default_graph()

print('Training Set Accuracy:', trainaccuracy, " (", len(train_y), " samples)")
print('Testing Set Accuracy:', testaccuracy, " (", len(test_y), " samples)")
print('F1 Score:', f1)
print('Confusion Martrix:\n', metrics.confusion_matrix(test_y, y_pred))