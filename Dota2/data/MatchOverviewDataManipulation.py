from sklearn import preprocessing, cross_validation, svm, neighbors,tree, metrics, linear_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('MatchOverview.csv', delimiter=",")

cols = df.shape[1]
X = df.iloc[:,1:cols-1]
y = df.iloc[:,cols-1:cols]

xdata = np.zeros((df.shape[0],226))              #111 col        df.shape[0] rows
ydata = np.zeros((df.shape[0],1))
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

np.savetxt('xdata.txt',xdata,delimiter=',')
np.savetxt('ydata.txt',ydata,delimiter=',')

