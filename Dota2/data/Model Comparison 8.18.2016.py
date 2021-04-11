from sklearn import preprocessing, cross_validation, svm, neighbors,tree, linear_model,metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tempfile import TemporaryFile
outfilex = TemporaryFile()
outfiley = TemporaryFile()

df = pd.read_csv('MatchOverview.csv', delimiter=",")

cols = df.shape[1]
X = df.iloc[:,1:cols-1]
y = df.iloc[:,cols-1:cols]

xdata = np.zeros((df.shape[0]*2,226))              #111 col        df.shape[0] rows
ydata = np.zeros((df.shape[0]*2,1))
j = df.shape[0]

for i in range(0,df.shape[0]):
    hero1 = X.iloc[i,0]
    hero2 = X.iloc[i,1]
    hero3 = X.iloc[i,2]
    hero4 = X.iloc[i,3]
    hero5 = X.iloc[i,4]
    hero6 = X.iloc[i,5]
    hero7 = X.iloc[i,6]
    hero8 = X.iloc[i,7]
    hero9 = X.iloc[i,8]
    hero10 = X.iloc[i,9]
    xdata[i, hero1-1] = 1
    xdata[i, hero2-1] = 1
    xdata[i, hero3-1] = 1
    xdata[i, hero4-1] = 1
    xdata[i, hero5-1] = 1
    xdata[i, hero6+112] = 1
    xdata[i, hero7+112] = 1
    xdata[i, hero8+112] = 1
    xdata[i, hero9+112] = 1
    xdata[i, hero10+112] = 1
    if y.iloc[i,0] == True:
        ydata[i,0] = 1
    else:
        ydata[i,0] = 2
    i=+1

X_train, X_test, y_train, y_test = cross_validation.train_test_split(xdata,ydata,test_size=0.19365,random_state=1)
y_train = np.ravel(y_train,order='C')

linclf = linear_model.Lasso(alpha=0.1)
#knearclf = neighbors.KNeighborsClassifier(n_jobs=-1)
rbfsvmclf = svm.SVC(C=1,kernel='rbf',max_iter=100000)
treeclf = tree.DecisionTreeClassifier()

linclf.fit(X_train, y_train)
#knearclf.fit(X_train, y_train)
rbfsvmclf.fit(X_train, y_train)
treeclf = treeclf.fit(X_train,y_train)

print(linclf.score(X_test,y_test))
#print(knearclf.score(X_test,y_test))
print(rbfsvmclf.score(X_test,y_test))

print(metrics.f1_score(y_test,linclf.predict(X_test)))
print(metrics.f1_score(y_test,rbfsvmclf.predict(X_test)))
print(metrics.f1_score(y_test,treeclf.predict(X_test)))
