import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('MasterBankData.txt')
df.replace('?', -99999, inplace=True)

X = np.array(df.drop(['result'],1))
y = np.array(df['result'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC(C=0.5,kernel='rbf',probability=False,decision_function_shape=None)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[34,6,3,2,0,147,1,0,1,6,5,151,2,-1,0], [55,2,2,2,0,147,1,1,0,6,3,151,1,-1,0]])

example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)