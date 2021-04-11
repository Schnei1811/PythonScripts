import numpy as np
import matplotlib.pyplot as plt
import os
import re
import random
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import neighbors, tree, svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
np.random.seed(3)

gaitfiles = os.listdir('WalkingActivity')
gaitdict = {}
gaitlenlist = []

for file in gaitfiles:
    gaitname = 'person' + file.split('.')[0]
    gaitdict[gaitname] = np.genfromtxt('WalkingActivity/{}'.format(file), delimiter=',')
    if np.isnan(gaitdict[gaitname]).any():
        print('{} has NAN data'.format(file))
    gaitlenlist.append(len(gaitdict[gaitname]))

print('\nHello! Let\'s walk through the process of identifying someone from their gait.\n'
      'First thing we want to do is load the csv files and check if our time series data for each individual '
      'are of similar length\n')

# print(math.isnan(gaitdict['person1']))

print('Min :', min(gaitlenlist))
print('Max :', max(gaitlenlist))

print('\nWoah! There\'s a huge disparity between the lengths of each sample. '
      '\nWe need to be aware of this moving forward or we could create a class imbalance\n'
      '\nLet\'s plot a box plot to help visualize the variation within our data')

plt.boxplot(gaitlenlist)
plt.xlabel('Gait Profiles')
plt.ylabel('Number of time series entries')
plt.title('Variation of time series entries')
plt.show()

print('\nHmm... So it\'s not just minimum and maximum. There\'s a large a large amount of variation throughout '
      'the dataset. \nThis will be important to keep in mind as we move forward'
      '\n\nNext, let\'s visualize what the data looks like by looking at the variation in the X, Y, Z '
      'signal expression for one individual')

x1 = gaitdict['person1'][:, 0]
y1 = gaitdict['person1'][:, 1]
y2 = gaitdict['person1'][:, 2]
y3 = gaitdict['person1'][:, 3]
plt.plot(x1, y1, label='X Response')
plt.plot(x1, y2, label='Y Response')
plt.plot(x1, y3, label='Z Response')
plt.xlabel('Seconds')
plt.ylabel('Signal Value')
plt.title('X, Y, and Z Signal Values for Individual 1')
plt.legend()
plt.show()

x1 = gaitdict['person1'][:, 0]
y1 = gaitdict['person1'][:, 1]
y2 = gaitdict['person1'][:, 2]
y3 = gaitdict['person1'][:, 3]
plt.plot(x1, y1, label='X Response')
plt.plot(x1, y2, label='Y Response')
plt.plot(x1, y3, label='Z Response')
plt.xlabel('Seconds')
plt.ylabel('Signal Value')
plt.title('X, Y, and Z Signal Values for Individual 1')
plt.legend()
plt.show()

x1 = gaitdict['person1'][:, 0]
y1 = gaitdict['person1'][:, 1]
y2 = gaitdict['person1'][:, 2]
y3 = gaitdict['person1'][:, 3]
plt.plot(x1, y1, label='X Response')
plt.plot(x1, y2, label='Y Response')
plt.plot(x1, y3, label='Z Response')
plt.xlabel('Seconds')
plt.ylabel('Signal Value')
plt.title('X, Y, and Z Signal Values for Individual 1')
plt.legend()
plt.show()

print('\nWe can see the signal expression are not the same but highly correlated. Data looks good visually.'
      '\n\nNow let\'s take a look at the variation between 3 individuals. To make the plot cleaner let\'s look'
      'at the X sensor only')

x1 = gaitdict['person1'][:, 0]
x2 = gaitdict['person2'][:, 0]
x3 = gaitdict['person3'][:, 0]
y1 = gaitdict['person1'][:, 1]
y2 = gaitdict['person2'][:, 1]
y3 = gaitdict['person3'][:, 1]
plt.plot(x1, y1, label='Individual 1')
plt.plot(x2, y2, label='Individual 2')
plt.plot(x3, y3, label='Individual 3')
plt.xlabel('Seconds')
plt.ylabel('X Response')
plt.title('Comparison of X Response of 3 Individuals')
plt.legend()
plt.show()

print('Well this looks promising! There are a few take aways here:'
      '\nWe can see an obvious difference between X signal per person which means classification should be possible.' 
      '\nWe can see overall the gait patterns are not uniform across an individual time series.'
      '\nWe can see visually the time series difference between samples we found earlier.')

print('\n\nSo how do we want to approach this problem? There is no specification for how to ID a person'
      'so let\'s make an objective ourselves!'
      '\nI think a good starting place is to determine if we can ID the people after 1 second of movement\n'
      '\nThat seems both ambitious and useful so let\'s see how we do.'
      '\nAlright well, how many time steps are 1 second? Are time responses even across individuals\n')

onesecondlist = []
for i in range(len(gaitfiles)):
    onesecondlist.append((np.abs(gaitdict['person{}'.format(i+1)][:, 0] - 1)).argmin())

stepsonesecond = int(round(sum(onesecondlist) / len(onesecondlist)))

print('Here\'s the index value for 1 second for each user. They\'re similar but not identical:\n')
print(onesecondlist)
print('\nBecause they\'re so close, let\'s take the rounded average of {} '
      'to equal 1 second\n'.format(stepsonesecond))

print('With the 1 second objective in mind, we are now ready to begin training our models\n')
print('We will need to create two data sets. A training set to train our classifier and a \n'
      'testing set to measure the performance of the model.')

train_data = []
test_data = []
poly = PolynomialFeatures(2, include_bias=False)

for key in gaitdict:
    lentimeseries = len(gaitdict[key])
    counter = 0
    individual = int(re.search(r'\d+', key).group()) - 1
    while True:
        if counter + stepsonesecond > lentimeseries:
            break
        oneseconddata = gaitdict[key][counter:counter + stepsonesecond, 1:]
        # oneseconddata = poly.fit_transform(oneseconddata)
        oneseconddata = [item for sublist in oneseconddata for item in sublist]
        counter += stepsonesecond
        if np.random.random() > 0.1:
            # while True:
            #     if np.random.random() > 1 - (lentimeseries / maxlen / stepsonesecond):
            #         break
            #     # print('hi', individual)
            #     train_data.append([np.random.normal(oneseconddata, 0.1), individual])
            train_data.append([oneseconddata, individual])
        else: test_data.append([oneseconddata, individual])


random.Random(1).shuffle(train_data)
random.Random(1).shuffle(test_data)

y_train = np.array([i[1] for i in train_data])
num_classifications = len(set(y_train))

unique, counts = np.unique(y_train, return_counts=True)
trainclassificationdict = dict(zip(unique, counts))

averagesamples = 0
for key in trainclassificationdict:
    averagesamples += trainclassificationdict[key]

averagesamples = averagesamples/num_classifications



x_train = np.array([i[0] for i in train_data])
y_train = np.array([i[1] for i in train_data])

x_test = np.array([i[0] for i in test_data])
y_test = np.array([i[1] for i in test_data])

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# print(averagesamples)

# print(x_train[0])
# print(np.random.normal(x_train[0], 0.1))
#
# print(x_train.shape)
# x_train = np.vstack((x_train, np.random.normal(x_train[0], 0.1)))
# print(x_train)


# plt.bar(trainclassificationdict.keys(), trainclassificationdict.values(), 1, color='g')
# plt.show()
#
# plt.bar(testclassificationdict.keys(), testclassificationdict.values(), 1, color='g')
# plt.show()




if not os.path.exists('VGG16_saved_models'):
    os.makedirs('VGG16_saved_models')

nnclf = neighbors.KNeighborsClassifier(n_neighbors=5)
nnclf.fit(x_train, y_train)
trainaccuracy = nnclf.score(x_train, y_train)
testaccuracy = nnclf.score(x_test, y_test)
f1 = metrics.f1_score(y_test, nnclf.predict(x_test), average='weighted')
with open('VGG16_saved_models/nnclf'.format(round(testaccuracy, 2)), 'wb') as f:
    pickle.dump(nnclf, f)

print(trainaccuracy)
print(testaccuracy)
print(f1)
print(confusion_matrix(y_test, nnclf.predict(x_test)))


rfclf = tree.DecisionTreeClassifier()
rfclf.fit(x_train, y_train)
trainaccuracy = rfclf.score(x_train, y_train)
testaccuracy = rfclf.score(x_test, y_test)
f1 = metrics.f1_score(y_test, rfclf.predict(x_test), average='weighted')
with open('VGG16_saved_models/rfclf'.format(round(testaccuracy, 2)), 'wb') as f:
    pickle.dump(rfclf, f)

print(trainaccuracy)
print(testaccuracy)
print(f1)
print(confusion_matrix(y_test, rfclf.predict(x_test)))

svmclf = svm.SVC(kernel='rbf', max_iter=-1)
svmclf.fit(x_train, y_train)
trainaccuracy = svmclf.score(x_train, y_train)
testaccuracy = svmclf.score(x_test, y_test)
f1 = metrics.f1_score(y_test, svmclf.predict(x_test), average='weighted')
with open('VGG16_saved_models/svmclf'.format(round(testaccuracy, 2)), 'wb') as f:
    pickle.dump(svmclf, f)

print(trainaccuracy)
print(testaccuracy)
print(f1)
print(confusion_matrix(y_test, svmclf.predict(x_test)))

logitclf = LogisticRegression()
logitclf.fit(x_train, y_train)
trainaccuracy = logitclf.score(x_test, y_test)
testaccuracy = logitclf.score(x_train, y_train)
f1 = metrics.f1_score(y_test, logitclf.predict(x_test), average='weighted')
with open('VGG16_saved_models/logitclf'.format(round(testaccuracy, 2)), 'wb') as f:
    pickle.dump(logitclf, f)

print(trainaccuracy)
print(testaccuracy)
print(f1)
print(confusion_matrix(y_test, logitclf.predict(x_test)))


fnnmodel = Sequential()
fnnmodel.add(Dense(1028, activation='relu', name='fc1', input_dim=len(x_train[0])))
fnnmodel.add(Dropout(0.2))
fnnmodel.add(Dense(len(set(y_test)), activation='softmax'))

fnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fnnmodel.fit(x_train, y_train_onehot, epochs=250, batch_size=32, validation_data=(x_test, y_test_onehot),
#           callbacks=[  # EarlyStopping(min_delta=0.001, patience=3),
#               ModelCheckpoint('VGG16_saved_models/FNN-{val_acc:.3f}.hdf5', monitor='val_acc',
#                               save_best_only=True)])

dnnmodel = Sequential()
dnnmodel.add(Dense(1028, activation='relu', name='fc1', input_dim=len(x_train[0])))
dnnmodel.add(Dropout(0.2))
dnnmodel.add(Dense(1028, activation='relu', name='fc2'))
dnnmodel.add(Dropout(0.2))
dnnmodel.add(Dense(len(set(y_test)), activation='softmax'))


dnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# model.fit(x_train, y_train_onehot, epochs=200, batch_size=32, validation_data=(x_test, y_test_onehot),
#           callbacks=[  # EarlyStopping(min_delta=0.001, patience=3),
#               ModelCheckpoint('VGG16_saved_models/DNN-{val_acc:.3f}.hdf5', monitor='val_acc',
#                               save_best_only=True)])

fnnmodel.load_weights('VGG16_saved_models/FNN-0.774.hdf5')
dnnmodel.load_weights('VGG16_saved_models/DNN-0.778.hdf5')


correct = 0
incorrectdict = {}

for i in range(len(test_data)):
    predictlist = [int(nnclf.predict(x_test[i:i+1])),
                   int(rfclf.predict(x_test[i:i+1])),
                   int(svmclf.predict(x_test[i:i + 1])),
                   int(logitclf.predict(x_test[i:i + 1])),
                   np.argmax(fnnmodel.predict(np.array([x_test[i]]))),
                   np.argmax(dnnmodel.predict(np.array([x_test[i]])))]
    if len(predictlist) > len(set(predictlist)):
        answer = np.argmax(dnnmodel.predict(np.array([x_test[i]])))
    else: answer = max(set(predictlist), key=predictlist.count)

    if answer == y_test[i]: correct += 1
    else:
        if y_test[i] not in incorrectdict: incorrectdict[y_test[i]] = 1
        else:
            if 'incorrectanswers' not in locals(): incorrectanswers = x_test[i]
            else: incorrectanswers = np.vstack((incorrectanswers, x_test[i]))
            incorrectdict[y_test[i]] += 1

print(incorrectdict)

print('Ensemble Accuracy: ', correct/len(y_test))


print('\nNearest Neighbour ', round(os.path.getsize('VGG16_saved_models/nnclf')/1000000, 2), 'MB')
print('Random Forest ', round(os.path.getsize('VGG16_saved_models/rfclf')/1000000, 2), 'MB')
print('Support Vector Machine ', round(os.path.getsize('VGG16_saved_models/svmclf')/1000000, 2), 'MB')
print('Logistic Regression ', round(os.path.getsize('VGG16_saved_models/logitclf')/1000000, 2), 'MB')
print('Simple Neural Network ', round(os.path.getsize('VGG16_saved_models/FNN-0.774.hdf5')/1000000, 2), 'MB')
print('Deep Neural Network ', round(os.path.getsize('VGG16_saved_models/DNN-0.778.hdf5')/1000000, 2), 'MB')

counter = 0
for key in gaitdict:
    while counter < len(gaitdict[key]):
        for i in range(len(x_test)):
            if np.array_equal(gaitdict[key][counter:counter + stepsonesecond, 1:], x_test[i].reshape(32, 3)):
                if 'tested' not in locals():
                    tested = gaitdict[key][counter:counter + stepsonesecond]
                    tested = np.vstack((tested, np.array([np.nan, np.nan, np.nan, np.nan])))
                else:
                    tested = np.vstack((tested, gaitdict[key][counter:counter + stepsonesecond]))
                    tested = np.vstack((tested, np.array([np.nan, np.nan, np.nan, np.nan])))
        for i in range(len(incorrectanswers)):
            if np.array_equal(gaitdict[key][counter:counter + stepsonesecond, 1:], incorrectanswers[i].reshape(32, 3)):
                if 'wrong' not in locals():
                    wrong = gaitdict[key][counter:counter + stepsonesecond]
                    wrong = np.vstack((wrong, np.array([np.nan, np.nan, np.nan, np.nan])))
                else:
                    wrong = np.vstack((wrong, gaitdict[key][counter:counter + stepsonesecond]))
                    wrong = np.vstack((wrong, np.array([np.nan, np.nan, np.nan, np.nan])))
        counter += 32
    try:
        x1 = gaitdict[key][:, 0]
        y1 = gaitdict[key][:, 1]
        x2 = tested[:, 0]
        y2 = tested[:, 1]
        x3 = wrong[:, 0]
        y3 = wrong[:, 1]
        plt.plot(x1, y1, color='black', label='Training Data')
        plt.plot(x2, y2, color='green', label='Testing Data Correctly Classified')
        plt.plot(x3, y3, color='red', label='Testing Data Incorrectly Classified')
        plt.xlabel('Seconds')
        plt.ylabel('Signal Value')
        plt.title('Individual {} X Axis Gait Data'.format(int(re.search(r'\d+', key).group()) - 1))
        plt.legend()
        plt.show()
    except:
        x1 = gaitdict[key][:, 0]
        y1 = gaitdict[key][:, 1]
        x2 = tested[:, 0]
        y2 = tested[:, 1]
        plt.plot(x1, y1, color='black', label='Training Data')
        plt.plot(x2, y2, color='green', label='Testing Data Correctly Classified')
        plt.xlabel('Seconds')
        plt.ylabel('Signal Value')
        plt.title('Individual {} X Axis Gait Data'.format(int(re.search(r'\d+', key).group()) - 1))
        plt.legend()
        plt.show()
    counter = 0
    del tested
    if 'wrong' in locals(): del wrong



















import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

# for key in gaitdict:
#     lentimeseries = len(gaitdict[key])
#     counter = 0
#     individual = int(re.search(r'\d+', key).group()) - 1
#     while True:
#         if counter + stepsonesecond > lentimeseries:
#             break
#         oneseconddata = gaitdict[key][counter:counter + stepsonesecond, 1:]
#         # oneseconddata = poly.fit_transform(oneseconddata)
#         # print(oneseconddata)
#         # oneseconddata = [item for sublist in oneseconddata for item in sublist]
#         counter += stepsonesecond
#         print(np.array(oneseconddata).shape)
#         all_data.append([np.array(oneseconddata), individual])
#         print(all_data[0][0].shape)
#
# print(all_data[0][0].shape)
#
# random.Random(12).shuffle(all_data)
#
# train_data = all_data[:int(0.9 * len(all_data))]
# test_data = all_data[int(0.9 * len(all_data)):]
#
# print(train_data[0][0].shape)
#
# x_train = np.array([i[0] for i in train_data])
# y_train = np.array([i[1] for i in train_data])
#
# print(x_train[0].shape)
#
# unique, counts = np.unique(y_train, return_counts=True)
# trainclassificationdict = dict(zip(unique, counts))
#
# x_test = np.array([i[0] for i in test_data])
# y_test = np.array([i[1] for i in test_data])

# print(x_train)
# print(y_test)

# def create_model(input_length):
#     print ('Creating model...')
#     model = Sequential()
#     # model.add(Embedding(input_dim = 188, output_dim = 1xdif-50))
#     model.add(LSTM(output_dim=256, activation='sigmoid', return_sequences=True, input_shape=len(x_train[0])))
#     model.add(Dropout(0.5))
#     model.add(LSTM(output_dim=256, activation='sigmoid'))
#     model.add(Dropout(0.5))
#     model.add(Dense(len(set(y_test)), activation='softmax'))
#
#     print ('Compiling...')
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     return model
#
# model = create_model(len(x_train[0]))
#
# y_train_onehot = to_categorical(y_train)
# y_test_onehot = to_categorical(y_test)
#
# print(x_train[0].shape)
# print(y_train_onehot.shape)
# print(y_test_onehot.shape)
#
# hist = model.fit(x_train, y_train_onehot, batch_size=8, nb_epoch=10, validation_data=(x_test, y_test_onehot))


































# #Visualize 1 person
# bins = np.arange(min(gaitlenlist), max(gaitlenlist), 20) # fixed bin size
# plt.hist(roundgaitlenlist, bins=bins, alpha=1, rwidth=2000)
# plt.title('Random Gaussian data (fixed bin size)')
# plt.xlabel('variable X (bin size = 5)')
# plt.ylabel('count')
#
# plt.show()
#
# x1 = gaitdict['person1'][:, 0]
# y1 = gaitdict['person1'][:, 1]
# y2 = gaitdict['person1'][:, 2]
# y3 = gaitdict['person1'][:, 3]
# plt.plot(x1, y1, label='X1 axis')
# plt.plot(x1, y2, label='X2 axis')
# plt.plot(x1, y3, label='X3 axis')
# plt.xlabel('Plot Number')
# plt.ylabel('Important var')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()
#
#
# #Compare 3 people walking.
#
# x1 = gaitdict['person1'][:, 0]
# x2 = gaitdict['person2'][:, 0]
# x3 = gaitdict['person3'][:, 0]
# y1 = gaitdict['person1'][:, 1]
# y2 = gaitdict['person2'][:, 1]
# y3 = gaitdict['person3'][:, 1]
# plt.plot(x1, y1, label='X1 axis')
# plt.plot(x2, y2, label='X2 axis')
# plt.plot(x3, y3, label='X3 axis')
# plt.plot(x4, y4, label='X4 axis')
# plt.plot(x5, y5, label='X5 axis')
# plt.xlabel('Plot Number')
# plt.ylabel('Important var')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()
#
# #Can visually see there are differences in X axis alone. This is good.
# #Problem. Data are not all the same length. Let's see what the data looks like
#
#
# diff = []
# for i in range(min(len(x1), len(x2))):
#     diff.append(y1[i] - y2[i])
#
# plt.plot(x2, diff)
# plt.xlabel('Plot Number')
# plt.ylabel('Important var')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()






