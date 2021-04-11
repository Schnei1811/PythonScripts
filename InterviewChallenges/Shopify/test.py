import re
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors, tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers

df = pd.read_excel(io='E-Bike_Survey_Responses.xls')
# print(df.head(5))

df = df.drop(columns='Timestamp')

columnlist = list(df)

print(list(df))

print(len(df))

uniquevaluedict = {}

for column in columnlist:
    uniquevaluedict[column] = len(df.groupby(column).nunique())

# print(uniquevaluedict)

df = df.fillna('0')

for column in columnlist:
    datadict = {}
    for index, row in df.iterrows():
        if row[column] not in datadict:
            datadict[row[column]] = 1
        else: datadict[row[column]] += 1

    for index, row in df.iterrows():
        if datadict[row[column]] <= 50:
            row[column] = '0'



# print(df['Does your household have access to any of the following types of private motorized vehicles?'])



columnlist = list(df)

# print(list(df))

# for i in range(len(columnlist)):
#     print(i, columnlist[i])

from sklearn.preprocessing import LabelEncoder


#Values:    0, 4, 7, 8
import time
number = LabelEncoder()
for i in range(len(columnlist)):
    if i in [0, 4, 7, 8, 10]: pass
    else:
        if 'data_matrix' not in locals():
            data_matrix = pd.get_dummies(df[columnlist[i]]).as_matrix()
            print(data_matrix)
        else:
            data_matrix = np.concatenate((data_matrix,
                            pd.get_dummies(df[columnlist[i]]).as_matrix()),
                            axis=1)

def determine_to_numeric_value(column):
    numericvalueslist = re.findall('\d+', row[column])
    if len(numericvalueslist) > 1: numericvalue = int(sum(map(int, numericvalueslist)) / len(numericvalueslist))
    else: numericvalue = int(numericvalueslist[0])
    return numericvalue


for index, row in df.iterrows():
    numericvalue = determine_to_numeric_value(columnlist[0])
    df.set_value(index, columnlist[0], numericvalue)
    numericvalue = determine_to_numeric_value(columnlist[4])
    df.set_value(index, columnlist[4], numericvalue)
    numericvalue = determine_to_numeric_value(columnlist[7])
    df.set_value(index, columnlist[7], numericvalue)

    row[columnlist[8]] = row[columnlist[8]].replace('1 hour', '60')
    row[columnlist[8]] = row[columnlist[8]].replace('commute', '0')

    numericvalueslist = re.findall('\d+', row[columnlist[8]])
    if len(numericvalueslist) > 1: numericvalue = int(sum(map(int, numericvalueslist)) / len(numericvalueslist))
    else: numericvalue = int(numericvalueslist[0])
    df.set_value(index, columnlist[8], numericvalue)



for i in range(len(columnlist)):
    if i in [0, 4, 7, 8]:

        print(df[columnlist[i]])

        data_matrix = np.concatenate((data_matrix,
                                      df[columnlist[i]].as_matrix().reshape(-1,1)), axis=1)

np.set_printoptions(threshold=np.nan)
print(data_matrix)

# print(df.head())

# print(df['Does your household have access to any of the following types of private motorized vehicles?'])

# 33

print(df[columnlist[10]])



X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != columnlist[10]],
                                                    df.loc[:, columnlist[10]],
                                                    test_size=0.1,
                                                    random_state=1)


nnclf = neighbors.KNeighborsClassifier(n_neighbors=5)
nnclf.fit(X_train, y_train)
trainaccuracy = nnclf.score(X_train, y_train)
testaccuracy = nnclf.score(X_test, y_test)
with open('VGG16_saved_models/nnclf', 'wb') as f:
    pickle.dump(nnclf, f)

print(trainaccuracy)
print(testaccuracy)

rfclf = tree.DecisionTreeClassifier()
rfclf.fit(X_train, y_train)
trainaccuracy = rfclf.score(X_train, y_train)
testaccuracy = rfclf.score(X_test, y_test)
with open('VGG16_saved_models/rfclf', 'wb') as f:
    pickle.dump(rfclf, f)

print(trainaccuracy)
print(testaccuracy)

svmclf = svm.SVC(kernel='rbf', max_iter=-1)
svmclf.fit(X_train, y_train)
trainaccuracy = svmclf.score(X_train, y_train)
testaccuracy = svmclf.score(X_test, y_test)
with open('VGG16_saved_models/svmclf', 'wb') as f:
    pickle.dump(svmclf, f)

print(trainaccuracy)
print(testaccuracy)

logitclf = LogisticRegression()
logitclf.fit(X_train, y_train)
trainaccuracy = logitclf.score(X_test, y_test)
testaccuracy = logitclf.score(X_train, y_train)
with open('VGG16_saved_models/logitclf', 'wb') as f:
    pickle.dump(logitclf, f)

print(trainaccuracy)
print(testaccuracy)


# print(y_train)
# print(y_test)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

print(y_train_onehot.shape, y_test_onehot.shape)

import numpy
numpy.set_printoptions(threshold=numpy.nan)
#
# print(y_train_onehot)
# print(y_test_onehot)

fnnmodel = Sequential()
fnnmodel.add(Dense(2048, activation='relu', name='fc1', kernel_initializer='glorot_normal',
                   input_dim=20))
fnnmodel.add(BatchNormalization())
fnnmodel.add(Dropout(0.2))
fnnmodel.add(Dense(y_train_onehot.shape[1], activation='softmax'))

fnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fnnmodel.fit(X_train, y_train_onehot, epochs=200, batch_size=64, validation_data=(X_test, y_test_onehot),
          callbacks=[ModelCheckpoint('VGG16_saved_models/FNN-{val_acc:.3f}.hdf5', monitor='val_acc',
                              save_best_only=True)])

dnnmodel = Sequential()
dnnmodel.add(Dense(2048, activation='relu', name='fc1', kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(0.1), input_dim=20))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.5))
dnnmodel.add(Dense(2048, activation='relu', name='fc2', kernel_initializer='glorot_normal',
                   kernel_regularizer=regularizers.l2(0.1)))
dnnmodel.add(Dropout(0.5))
dnnmodel.add(Dense(y_train_onehot.shape[1], activation='softmax'))


dnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

dnnmodel.fit(X_train, y_train_onehot, epochs=200, batch_size=64, validation_data=(X_test, y_test_onehot),
          callbacks=[ModelCheckpoint('VGG16_saved_models/DNN-{val_acc:.3f}.hdf5', monitor='val_acc',
                              save_best_only=True)])



# fnnmodel.load_weights('VGG16_saved_models/FNN-0.652.hdf5')
# dnnmodel.load_weights('VGG16_saved_models/DNN-0.647.hdf5')

correct = 0
incorrectdict = {}

from scipy import stats

for i in range(len(X_test)):

    predictlist = [#int(nnclf.predict(X_test[i:i+1])),
                   # int(rfclf.predict(X_test[i:i+1])),
                   int(svmclf.predict(X_test[i:i + 1])),
                   int(logitclf.predict(X_test[i:i + 1])),
                   np.argmax(fnnmodel.predict(np.array(X_test[i:i + 1]))),
                   np.argmax(dnnmodel.predict(np.array(X_test[i:i + 1])))]

    # print(predictlist)


    answer = max(set(predictlist), key=predictlist.count)
    # answer = np.argmax(fnnmodel.predict(np.array(X_test[i:i + 1])))

    # print(answer)

    # print(predictlist)
    # print(len(predictlist), len(set(predictlist)))

    if answer == int(y_test[i:i+1]): correct += 1

print(correct/len(X_test)*100)












