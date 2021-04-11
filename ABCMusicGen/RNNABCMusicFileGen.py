#import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree
np.set_printoptions(threshold=np.nan)
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import numpy as np
from numpy import genfromtxt
import sys

# % must be converted back to =
# . must be converted back to ,

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

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

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	global snnepoch
	m = X.shape[0]
	X = np.matrix(X)
	y = np.matrix(y)
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	J = 0
	delta1 = np.zeros(theta1.shape)
	delta2 = np.zeros(theta2.shape)
	for i in range(m):
		try:
			first_term = np.multiply(-y[i,:], np.log(h[i,:]))
		except ValueError:
			print("Check if you have classifications that are 0 or if num_labels is correct")
		second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
		J += np.sum(first_term - second_term)
	J = J / m
	J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
	for t in range(m):
		a1t = a1[t,:]
		z2t = z2[t,:]
		a2t = a2[t,:]
		ht = h[t,:]
		yt = y[t,:]
		d3t = ht - yt
		z2t = np.insert(z2t, 0, values=np.ones(1))
		d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
		delta1 = delta1 + (d2t[:,1:]).T * a1t
		delta2 = delta2 + d3t.T * a2t
	delta1 = delta1 / m
	delta2 = delta2 / m
	delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
	delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
	grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
	#print("Iteration:",snnepoch, " Cost:",J)
	snnepoch = snnepoch +1
	return J, grad

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	m = X.shape[0]
	X = np.matrix(X)
	y = np.matrix(y)
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	J = 0
	for i in range(m):
		first_term = np.multiply(-y[i,:], np.log(h[i,:]))
		second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
		J += np.sum(first_term - second_term)
	J = J / m
	J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
	return J

def isPrime(n):     #Return True if Prime
	if n == 2:
		return True
	if n == 3:
		return True
	if n % 2 == 0:
		return False
	if n % 3 == 0:
		return False
	i = 5
	w = 2
	while i * i <= n:
		if n % i == 0:
			return False
		i += w
		w = 6 - w
	return True

def recurrent_neural_network(dataname,x,n_classes,n_features,rnn_size,n_chunks,chunk_size):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
		 'biases':tf.Variable(tf.random_normal([n_classes]))}
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)
	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
	return output

def CreateDictionary():
	outputscale, inputscale = {},{}
	i = 0
	n = 4
	while i < df.shape[0]:
		while n < LengthMusic+5:
			s = df['%s'%str(n)]
			note = s.iloc[i]
			try:
				foo = inputscale[note]
			except KeyError:
				outputscale[len(outputscale)] = note
				inputscale = {v: k for k, v in outputscale.items()}
			n = n + 1
		n = 4
		i = i + 1
	print(outputscale)
	print(len(outputscale))
	return outputscale, inputscale

def TrainRandomForest(newsong_data):
	cols = data.shape[1]
	X = data[:, 0:cols - 1]
	y = data[:, cols - 1:cols]
	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
	train_y = np.ravel(train_y)
	rfclf.fit(train_x,train_y)
	newsong_data = newsong_data.reshape(1,-1)
	intnewnote = int(rfclf.predict(newsong_data))
	newnote = outputscale[int(rfclf.predict(newsong_data))]
	newsong[len(newsong)] = newnote
	newsong_data = np.append(newsong_data,np.zeros((1,UniqueCharacter)))
	newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
	#print(newsong)
	#print(round(startingnote / LengthMusic,2),"%")
	return newsong_data

def TrainSimpleNN(newsong_data,ynoteonehot):
	cols = data.shape[1]
	X = data[:, 0:cols - 1]
	y = data[:, cols - 1:cols]

	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

	input_size = train_x.shape[1]
	hidden_size = 500
	regularization = 1
	maxiter = 50
	encoder = OneHotEncoder(sparse=False)
	num_labels = len(np.unique(y))

	train_y_onehot = encoder.fit_transform(train_y)
	y_onehot = np.matrix(train_y_onehot)
	X = np.matrix(train_x)

	params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
	print("Training...")
	fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, regularization),
				method='TNC', jac=True, options={'maxiter': maxiter})
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	X = np.matrix(newsong_data)
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	intnewnote = int(np.array(np.argmax(h, axis=1) + 1))
	print(intnewnote)
	intnewnote = np.unique(y)[intnewnote-1]
	newnote = outputscale[intnewnote]
	newsong[len(newsong)] = newnote
	newsong_data = np.append(newsong_data, np.zeros((1, UniqueCharacter)))
	newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
	print(newsong)
	global snnepoch
	snnepoch = 0
	return newsong_data

def TrainRecurrentNN(newsong_data,dataname):
	cols = data.shape[1]
	X = data[:, 0:cols - 1]
	y = data[:, cols - 1:cols]

	train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
	train_x, test_x, train_y, test_y = train_x.astype(int), test_x.astype(int), train_y.astype(int), test_y.astype(int)

	encoder = OneHotEncoder(sparse=False)
	train_y_onehot = np.matrix(encoder.fit_transform(train_y))
	test_y_onehot = np.matrix(encoder.fit_transform(test_y))

	num_labels = len(np.unique(y))
	n_features = len(X[0])
	hm_epochs = 100

	featuresprimecounter = 0
	featuresPrime, lenPrime = isPrime(n_features), isPrime(train_x.shape[0])
	if lenPrime == True:
		train_x = train_x[0:train_x.shape[0]-1,:]
	if featuresPrime == True:
		print('prime')
		featuresprimecounter = 1
		zeros = np.zeros((train_x.shape[0],1))
		train_x = np.append(train_x,zeros, axis=1)
		newsong_data = np.insert(newsong_data,len(newsong_data),0)

	i = 256
	for i in range(256,0,-1):
		if len(train_x[:]) % i == 0:
			batch_size = i
			break
	i = 0
	for i in range(round((n_features+featuresprimecounter)**0.5),1,-1):
		if (n_features+featuresprimecounter) % i == 0:
			chunk_size = i
			n_chunks = int((n_features+featuresprimecounter) / chunk_size)              #nfeatures must be evenly divisible by batch size.  #chunk size and nchunks must multiply to nfeatures
			break

	rnn_size = 64                                              #Increase for increased performance

	rnnx = tf.placeholder('float', [None, n_chunks,chunk_size])
	rnny = tf.placeholder('float')

	prediction = recurrent_neural_network(dataname,rnnx,num_labels,n_features,rnn_size,n_chunks,chunk_size)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,rnny))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(int(hm_epochs)):
			epoch_loss = 0
			i = 0
			while i<len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))
				batch_y = np.array(train_y_onehot[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={rnnx: batch_x, rnny: batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		saver.save(sess, "RNN/%sRNN.ckpt" %dataname)
		y_p = tf.argmax(prediction, 1)
		input_data = newsong_data.reshape((n_chunks,chunk_size))
		result = (sess.run(tf.argmax(prediction.eval(feed_dict={rnnx:[input_data]}),1)))
	intnewnote = int(result)
	intnewnote = np.unique(y)[intnewnote-1]
	newnote = outputscale[intnewnote]
	newsong[len(newsong)] = newnote
	newsong_data = np.append(newsong_data, np.zeros((1, UniqueCharacter)))
	newsong_data[lendescriptors + UniqueCharacter * startingnote + intnewnote] = 1
	print(newsong)
	tf.reset_default_graph()
	if featuresPrime == True:
		newsong_data = newsong_data[0:-1]
	return newsong_data

rfclf = tree.DecisionTreeClassifier()
snnepoch = 0

outputtimemetric = {1:'4/4',2:'3/4',3:'1/4',4:'2/4',5:'6/8'}
outputkey = {1:'Cmaj'}
outputsetting = {1:'Field',2:'Town',3:'Desert',4:'IceWorld',5:'Battle',6:'Boss'}
outputmood = {1:'Happy',2:'Sad'}

inputtimemetric = {v: k for k, v in outputtimemetric.items()}
inputkey = {v: k for k, v in outputkey.items()}
inputsetting = {v: k for k, v in outputsetting.items()}
inputmood = {v: k for k, v in outputmood.items()}

lendescriptors = 1 + len(outputtimemetric) + len(outputkey)

#df = pd.read_csv('ReplacedTownMaster200x600.csv')
#df = pd.read_csv('DrWily.csv')
#df = pd.read_csv('FFTown1.csv')
df = pd.read_csv('DrWilyFFTown1Mix.csv')
#df = pd.read_csv('WaterTownFFTown2Mix.csv')
#df = pd.read_csv('DrWilyPalaceMix.csv')
LengthMusic = df.shape[1]-4                     #4 = 0, Tempo, TimeMetric, Key

outputscale, inputscale = CreateDictionary()

UniqueCharacter = len(inputscale)
num_features = UniqueCharacter * LengthMusic
input_data = np.zeros(shape=(df.shape[0],num_features+lendescriptors),dtype=np.int)
Timing = df['1']
Tempo = df['2']
Key = df['3']

#Build OneHot DataSet: input_data
i = 0
n = 0
while i < df.shape[0]:
	input_data[i,0] = Tempo[i]
	input_data[i,inputtimemetric[Timing[i]]] = 1
	input_data[i,inputkey[Key[i]]+5] = 1
	k = 0
	while n < LengthMusic:
		s = df['%s'%str(n+4)]
		note = s.iloc[i]
		input_data[i,k+inputscale[note]+lendescriptors]=1
		n = n + 1
		k = k + UniqueCharacter
	n=0
	i = i + 1

#Create Training Data
songtempo, songtiming, songkey = 175, '4/4', 'Cmaj'
note1, note2, note3 = '+f+', 'g', 'a'
#songtempo, songtiming, songkey = 140, '4/4', 'Cmaj'
#note1, note2, note3 = '+f+', 'c', 'c'
#songtempo, songtiming, songkey = 120, '4/4', 'Cmaj'
#note1, note2, note3 = '+fff+', 'z2', 'z3/4'
outputstring = note1 + ' ' + note2 + ' ' + note3
newsong = {0:note1,1:note2,2:note3}
songtiming = inputtimemetric[songtiming]
songkey = inputkey[songkey]
note1 = inputscale[note1]
note2 = inputscale[note2]
note3 = inputscale[note3]
notes = {0:note1,1:note2,2:note3}
startingnote = 3
LenghthMusic = 50
newsong_data = np.zeros((1, lendescriptors + UniqueCharacter * startingnote))
newsong_data[0, 0] = songtempo
newsong_data[0, songtiming] = 1
newsong_data[0, len(outputtimemetric) + songkey] = 1
newsong_data[0, lendescriptors + UniqueCharacter * 0 + notes[0]] = 1
newsong_data[0, lendescriptors + UniqueCharacter * 1 + notes[1]] = 1
newsong_data[0, lendescriptors + UniqueCharacter * 2 + notes[2]] = 1
LengthMusic = 50
while startingnote < LengthMusic:
	y = np.zeros((df.shape[0],1))
	X = input_data[:,0:lendescriptors+UniqueCharacter*startingnote]
	ynoteonehot = input_data[:,lendescriptors+UniqueCharacter*startingnote:lendescriptors+(startingnote*UniqueCharacter)+UniqueCharacter]
	n = 0
	i = 0
	while i < df.shape[0]:
		while n < UniqueCharacter:
			if ynoteonehot[i,n] == 1:
				newvar = n / UniqueCharacter
				note = inputscale[outputscale[round((newvar % 1) * UniqueCharacter)]]
				y[i] = int(note)
			n = n + 1
		n = 0
		i = i + 1
	data = np.append(X,y, axis=1)
	dataname = 'note%s'%startingnote
	#newsong_data = TrainRandomForest(newsong_data)
	#newsong_data = TrainSimpleNN(newsong_data,ynoteonehot)
	newsong_data = TrainRecurrentNN(newsong_data,dataname)
	startingnote = startingnote + 1

OutTempo = newsong_data[0]

n = 1
while n < len(outputtimemetric)+1:
	if newsong_data[n] == 1:
		OutTiming = outputtimemetric[n]
	n = n + 1

n = len(outputtimemetric)
while n < len(outputtimemetric)+len(inputkey)+1:
	if newsong_data[n] == 1:
		OutKey = outputkey[n-len(outputtimemetric)]
	n = n + 1

inputnewsong = {v: k for k, v in outputscale.items()}
print(newsong)
i = 3
while i < len(newsong):
	outputstring = outputstring + ' ' + newsong[i]
	i = i + 1

outputstring = outputstring.replace("%","=")
outputstring = outputstring.replace(".",",")

text_file = open("Test.txt",'w')
text_file.write("M: " + str(OutTiming) + "\nQ: " + str(int(OutTempo)) + "\nK: " + str(OutKey) + "\n\n" + outputstring)
text_file.close()