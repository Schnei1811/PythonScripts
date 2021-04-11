import re
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors, tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import operator

df = pd.read_excel(io='E-Bike_Survey_Responses.xls')
#

df = df.drop(columns='Timestamp')

columnlist = list(df)

print(list(df))

print(len(df))

uniquevaluedict = {}

for column in columnlist:
    uniquevaluedict[column] = len(df.groupby(column).nunique())


nan_dict = {}

for column in list(df):
    nan_dict[column[:70]] = int(len(df[column]) - df[column].count())

sorted_nan_dict = sorted(nan_dict.items(), key=operator.itemgetter(1), reverse=True)

labels, values = zip(*sorted_nan_dict)
indexes = np.arange(len(labels))


plt.title('NaN Entries Per Feature')
plt.bar(nan_dict.keys(), nan_dict.values(), 1, color='g')
plt.xticks(rotation='vertical')
plt.show()

# print(uniquevaluedict)

df = df.fillna('0')

for column in columnlist:
    datadict = {}
    for index, row in df.iterrows():
        if row[column] not in datadict:
            datadict[row[column]] = 1
        else: datadict[row[column]] += 1

    for index, row in df.iterrows():
        if datadict[row[column]] <= 100:
            row[column] = '0'



print(df['Does your household have access to any of the following types of private motorized vehicles?'])



columnlist = list(df)

# print(list(df))

# for i in range(len(columnlist)):
#     print(i, columnlist[i])



#Values:    0, 4, 7, 8


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
    row[columnlist[0]] = row[columnlist[0]].replace('younger', '10')
    row[columnlist[0]] = row[columnlist[0]].replace('more', '80')

    numericvalue = determine_to_numeric_value(columnlist[0])
    df.set_value(index, columnlist[0], numericvalue)

    row[columnlist[4]] = row[columnlist[4]].replace('Under', '0')
    row[columnlist[4]] = row[columnlist[4]].replace('100', '150')

    numericvalue = determine_to_numeric_value(columnlist[4])
    df.set_value(index, columnlist[4], numericvalue)

    row[columnlist[7]] = row[columnlist[7]].replace('Under', '0')
    row[columnlist[7]] = row[columnlist[7]].replace('Over', '50')

    numericvalue = determine_to_numeric_value(columnlist[7])
    df.set_value(index, columnlist[7], numericvalue)

    row[columnlist[8]] = row[columnlist[8]].replace('1 hour', '75')
    row[columnlist[8]] = row[columnlist[8]].replace('commute', '0')

    numericvalue = determine_to_numeric_value(columnlist[8])
    df.set_value(index, columnlist[8], numericvalue)


for i in range(len(columnlist)):
    if i in [0, 4, 7, 8]:

        print(df[columnlist[i]])

        data_matrix = np.concatenate((data_matrix,
                                      df[columnlist[i]].as_matrix().reshape(-1,1)), axis=1)

# np.set_printoptions(threshold=np.nan)
# print(data_matrix)

# print(df.head())

# print(df['Does your household have access to any of the following types of private motorized vehicles?'])

# 33

print(df[columnlist[10]])

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()

number.fit(df[columnlist[10]])
df[columnlist[10]] = number.transform(df[columnlist[10]])

print(df[columnlist[10]])


X_train, X_test, y_train, y_test = train_test_split(data_matrix,
                                                    df.loc[:, columnlist[10]],
                                                    test_size=0.1,
                                                    random_state=1)


nnclf = neighbors.KNeighborsClassifier(n_neighbors=50)
nnclf.fit(X_train, y_train)
trainaccuracy = nnclf.score(X_train, y_train)
testaccuracy = nnclf.score(X_test, y_test)
with open('VGG16_saved_models/nnclf', 'wb') as f:
    pickle.dump(nnclf, f)

print(trainaccuracy)
print(testaccuracy)
print(confusion_matrix(y_test, nnclf.predict(X_test)))

rfclf = tree.DecisionTreeClassifier()
rfclf.fit(X_train, y_train)
trainaccuracy = rfclf.score(X_train, y_train)
testaccuracy = rfclf.score(X_test, y_test)
with open('VGG16_saved_models/rfclf', 'wb') as f:
    pickle.dump(rfclf, f)

print(trainaccuracy)
print(testaccuracy)
print(confusion_matrix(y_test, rfclf.predict(X_test)))

svmclf = svm.SVC(kernel='rbf', max_iter=-1)
svmclf.fit(X_train, y_train)
trainaccuracy = svmclf.score(X_train, y_train)
testaccuracy = svmclf.score(X_test, y_test)
with open('VGG16_saved_models/svmclf', 'wb') as f:
    pickle.dump(svmclf, f)

print(trainaccuracy)
print(testaccuracy)
print(confusion_matrix(y_test, svmclf.predict(X_test)))

logitclf = LogisticRegression()
logitclf.fit(X_train, y_train)
trainaccuracy = logitclf.score(X_test, y_test)
testaccuracy = logitclf.score(X_train, y_train)
with open('VGG16_saved_models/logitclf', 'wb') as f:
    pickle.dump(logitclf, f)

print(trainaccuracy)
print(testaccuracy)
print(confusion_matrix(y_test, logitclf.predict(X_test)))

# print(y_train)
# print(y_test)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

print(y_train_onehot.shape, y_test_onehot.shape)



# print(y_train_onehot)
# print(y_test_onehot)

print(data_matrix.shape)

fnnmodel = Sequential()
fnnmodel.add(Dense(2048, activation='relu', name='fc1', kernel_initializer='glorot_normal',
                   kernel_regularizer=regularizers.l2(0.001), input_dim=data_matrix.shape[1]))
fnnmodel.add(BatchNormalization())
fnnmodel.add(Dropout(0.2))
fnnmodel.add(Dense(y_train_onehot.shape[1], activation='softmax'))

fnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fnnmodel.fit(X_train, y_train_onehot, epochs=200, batch_size=32, validation_data=(X_test, y_test_onehot),
          callbacks=[ModelCheckpoint('VGG16_saved_models/FNN-{val_acc:.3f}.hdf5', monitor='val_acc',
                              save_best_only=True)])

dnnmodel = Sequential()
dnnmodel.add(Dense(2048, activation='relu', name='fc1', kernel_initializer='glorot_normal',
                    kernel_regularizer=regularizers.l2(0.001), input_dim=data_matrix.shape[1]))
dnnmodel.add(BatchNormalization())
fnnmodel.add(Dropout(0.2))
dnnmodel.add(Dense(2048, activation='relu', name='fc2', kernel_initializer='glorot_normal',
                   kernel_regularizer=regularizers.l2(0.001)))
dnnmodel.add(BatchNormalization())
fnnmodel.add(Dropout(0.2))
dnnmodel.add(Dense(y_train_onehot.shape[1], activation='softmax'))


dnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

dnnmodel.fit(X_train, y_train_onehot, epochs=200, batch_size=32, validation_data=(X_test, y_test_onehot),
          callbacks=[ModelCheckpoint('VGG16_saved_models/DNN-{val_acc:.3f}.hdf5', monitor='val_acc',
                              save_best_only=True)])


fnnmodel.load_weights('VGG16_saved_models/FNN-0.661.hdf5')
dnnmodel.load_weights('VGG16_saved_models/DNN-0.674.hdf5')

correct = 0
incorrectdict = {}
correctNo = 0
correctNototal = 0

print(y_test_onehot.shape)

for i in range(len(X_test)):

    predictlist = [int(nnclf.predict(X_test[i:i+1])),
                   int(rfclf.predict(X_test[i:i+1])),
                   int(svmclf.predict(X_test[i:i + 1])),
                   int(logitclf.predict(X_test[i:i + 1])),
                   np.argmax(fnnmodel.predict(np.array(X_test[i:i + 1]))),
                   np.argmax(dnnmodel.predict(np.array(X_test[i:i + 1])))]

    if len(set(predictlist)) == 3:
        answer = np.argmax(dnnmodel.predict(np.array(X_test[i:i + 1])))
    else: answer = max(set(predictlist), key=predictlist.count)

    if int(y_test[i:i+1]) == 1 and answer == 1:
        correctNo += 1
        correctNototal += 1
    elif int(y_test[i:i+1]) == 1 and answer != 1:
        correctNototal += 1
    if answer == int(y_test[i:i+1]): correct += 1

print(correct/len(X_test)*100)
print(correctNo/correctNototal*100)
print(correctNo, correctNototal)











