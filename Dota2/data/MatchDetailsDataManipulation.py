import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import sys
import pickle
from sklearn import preprocessing, cross_validation,svm, neighbors,tree
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import time

df = pd.read_csv('MatchDetail.csv', delimiter=",")
cols = df.shape[1]
X = df.iloc[:,1:cols]

xdata = np.zeros((df.shape[0],1))

for i in range(0,df.shape[0]):
    team = X.iloc[i,1]
    if team >= 5:
        xdata[i, 0] = 2
    else:
        xdata[i, 0] = 1

FullData = np.concatenate([X,xdata],axis=1)         #Data minus game# plus team side
HeroAvgs = np.zeros((113,22))
HeroCount = np.zeros((113,22))

for i in range (0,113):
    HeroAvgs[i,0] = i+1

for _ in range(0,22):
    for i in range(0,114):
        temp2 = 0
        for j in range(0,df.shape[0]):
            if FullData[j,0] == i:
                if FullData[j,_] != 0:
                    temp = FullData[j,_]
                    temp2 = temp2 + temp
                    HeroAvgs[i-1, _] = temp2
                    HeroCount[i - 1, _] = HeroCount[i - 1, _] + 1

print(HeroCount)

np.savetxt('HeroCount',HeroCount,delimiter=',')



for _ in range(0,22):
    for i in range(0,114):
        HeroAvgs[i-1,_] = HeroAvgs[i-1,_]/HeroCount[i-1,_]

print(HeroAvgs)
print(HeroAvgs[50])

np.savetxt('HeroAvgs2',HeroAvgs,delimiter=',')












