# K-nearest neighbours. Want to always make K is appropriately sized for the problem
# 2 groups: K = 3 otherwise could have a split vote
# 3 groups: K = 5 otherwise could have a split vote
# Could also code in a random result when split
# Can determine confidence (ex. a 0 0 1 split classification = 66% confidence)
# Larger the dataset, the worse this algorithm runs
# Euclidean Distance -> Measures a point against each other point of the dataset

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('KNNcancerdata')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,10,3,6,2,3,5,4,10],[4,10,3,5,1,10,5,3,10],[4,8,3,5,4,5,10,1,6]])

example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)