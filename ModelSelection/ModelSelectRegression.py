import os
import tensorflow as tf
import numpy as np
import pickle
import time
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn import linear_model

if not os.path.exists("./regressiondata"):
    print("\nA regressiondata folder has been created in this directory.\nPlease place a supervised learning dataset in this folder. All outputs must be in the far right columns.\n")
    os.makedirs("./regressiondata")

def TrainLinearRegression(train_x, test_x, train_y, test_y):
    linreg = linear_model.LinearRegression()
    linreg.fit(train_x, train_y)
    Rsquared = linreg.score(test_x, test_y)
    print(Rsquared)

def TrainRidgeRegression(train_x, test_x, train_y, test_y):
    ridgereg = linear_model.RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10, 100])
    ridgereg.fit(train_x, train_y)
    Rsquared = ridgereg.score(test_x, test_y)
    print(Rsquared)

def TrainLassoRegression(train_x, test_x, train_y, test_y):
    lassoreg = linear_model.LassoCV(n_alphas=100, max_iter = -1)
    train_y = train_y.ravel()
    lassoreg.fit(train_x, train_y)
    Rsquared = lassoreg.score(test_x, test_y)
    print(Rsquared)

def TrainElasticNetRegression(train_x, test_x, train_y, test_y):
    elasticreg = linear_model.ElasticNetCV(n_alphas=100, max_iter = -1)
    train_y = train_y.ravel()
    elasticreg.fit(train_x, train_y)
    Rsquared = elasticreg.score(test_x, test_y)
    print(Rsquared)

def TrainLassoLarsRegression(train_x, test_x, train_y, test_y):
    lassolarsreg = linear_model.LassoLars(alpha=1.0, max_iter = -1)
    lassolarsreg.fit(train_x, train_y)
    Rsquared = lassolarsreg.score(test_x, test_y)
    print(Rsquared)

def TrainSGDRegressor(train_x, test_x, train_y, test_y):
    SGDreg = linear_model.SGDRegressor(alpha=0.01, n_iter = 1000)
    train_y = train_y.ravel()
    SGDreg.fit(train_x, train_y)
    Rsquared = SGDreg.score(test_x, test_y)
    print(Rsquared)

def TrainLinearSVR(train_x, test_x, train_y, test_y):
    linsvrreg = svm.LinearSVR()
    train_y = train_y.ravel()
    linsvrreg.fit(train_x, train_y)
    Rsquared = linsvrreg.score(test_x, test_y)
    print(Rsquared)

def TrainRBFSVR(train_x, test_x, train_y, test_y):
    rbfsvr = svm.SVR()
    train_y = train_y.ravel()
    rbfsvr.fit(train_x, train_y)
    try:
        Rsquared = rbfsvr.score(test_x, test_y)
        print(Rsquared)
    except ZeroDivisionError: print('Rsquared Divided by zero')

def TrainKNNRegressor(train_x, test_x, train_y, test_y):
    knnreg = neighbors.KNeighborsRegressor(weights='distance')
    knnreg.fit(train_x, train_y)
    Rsquared = knnreg.score(test_x, test_y)
    print(Rsquared)

def TrainRadiusKNNRegressor(train_x, test_x, train_y, test_y):
    radknnreg = neighbors.RadiusNeighborsRegressor(weights='distance')
    radknnreg.fit(train_x, train_y)
    Rsquared = radknnreg.score(test_x, test_y)
    print(Rsquared)

def TrainDecisionTreeRegressor(train_x, test_x, train_y, test_y):
    treereg = tree.DecisionTreeRegressor()
    treereg.fit(train_x, train_y)
    Rsquared = treereg.score(test_x, test_y)
    print(Rsquared)

def TrainGradientBoostingRegressor(train_x, test_x, train_y, test_y):
    boostreg = ensemble.GradientBoostingRegressor()
    train_y = train_y.ravel()
    boostreg.fit(train_x, train_y)
    Rsquared = boostreg.score(test_x, test_y)
    print(Rsquared)

def ModelSelect():
    ans = input('\nTrain which model? (Linear, Lars, SGD, SVR, KNN, Tree, Boost, or help, exit): ')
    if ans in ['Linear']:
        ans = input('\nAdd Regulaization? (No, Ridge, Lasso, ElasticNet, or help, exit): ')
        if ans in ['No']: TrainLinearRegression(train_x, test_x, train_y, test_y)
        elif ans in ['Ridge']: TrainRidgeRegression(train_x, test_x, train_y, test_y)
        elif ans in ['Lasso']: TrainLassoRegression(train_x, test_x, train_y, test_y)
        elif ans in ['ElasticNet']: TrainElasticNetRegression(train_x, test_x, train_y, test_y)
        elif ans in ['help']:
            print('\nRidge Regression: Linear regression with regularization to penalize large sized coefficients'
              '\nLasso Regression: Linear regression which estimates sparse coefficients. Prefers solution with fewer parameters'
              '\nElastic Net Regression: Linear regression combining Ridge and Lasso. Useful when multiple features are correlated to each other')
        elif ans in ['exit']: RepeatedModelSelect()
        else: print('Improper Selection'), RepeatedModelSelect()
    elif ans in ['Lars']:
        if np.where(X<0): print('Lars does not work with negative input values'), RepeatedModelSelect()
        TrainLassoLarsRegression(train_x, test_x, train_y, test_y)
    elif ans in ['SGD']: TrainSGDRegressor(train_x, test_x, train_y, test_y)
    elif ans in ['SVR']:
        ans = input('\nVersion? (Linear, RBF, or help, exit): ')
        if ans in ['Linear']: TrainLinearSVR(train_x, test_x, train_y, test_y)
        elif ans in ['RBF']: TrainRBFSVR(train_x, test_x, train_y, test_y)
        elif ans in ['help']:
            print('\nLinear SVR: Linear regression with regularization to penalize large sized coefficients'
                  '\nRadial Basis Function Support Vector Regression: Linear regression which estimates sparse coefficients. Prefers solution with fewer parameters')
        elif ans in ['exit']: RepeatedModelSelect()
        else: print('Improper Selection'), RepeatedModelSelect()
    elif ans in ['KNN']:
        ans = input('\nVersion? (Distance, Radius, or help, exit): ')
        if ans in ['Distance']: TrainKNNRegressor(train_x, test_x, train_y, test_y)
        elif ans in ['Radius']: TrainRadiusKNNRegressor(train_x, test_x, train_y, test_y)
        elif ans in ['help']:
            print('\nK Nearest Neighbour Regressor: Compares distance from each point'
                  '\nRadius K Nearest Neighbour Regressor: Compares distance based on a fixed radius r')
    elif ans in ['Tree']: TrainDecisionTreeRegressor(train_x, test_x, train_y, test_y)
    elif ans in ['Boost']: TrainGradientBoostingRegressor(train_x, test_x, train_y, test_y)
    elif ans in ['help']:
        print('\nLinear Regression: Minimizes the residual sum of squares of observed response from Data and predicted response from linear approximation'
              '\nLasso Lars Regression: Least-angle regression with lasso regularization. Good for high-dimension Data. Sensitive to noise'
              '\nStochastic Gradient Descent Regression: Gradient of loss is estimated each sample and updated by learning rate'
              '\nSupport Vector Regression: Effective in high dimensional space'
              '\nNearest Neighbour Regression: Each point in the local neighbourhood contributes to the classification of query point'
              '\nTree Regression: Tree'
              '\nGradient Boosting: Ensemble model')
    elif ans in ['exit']: exit()
    else: print('Improper Selection'), RepeatedModelSelect()

def RepeatedModelSelect():
    while True:
        ans = input("\nTrain Data with Alternative Model? (y/n): ")
        if ans in ["y"]: ModelSelect()
        elif ans in ['n']: exit()
        else: print('Improper Command'), RepeatedModelSelect()

def LoadData():
    dirs = os.listdir("regressiondata/")
    print("\nAvailable Datasets: \n")
    for file in dirs: print(file)
    datarun = input("\nEnter a Dataset: ")
    return datarun

def CheckDataName():
    while True:
        datarun = LoadData()
        dataname = datarun[:-4]
        try:
            DataSet = np.loadtxt('regressiondata/{}'.format(datarun), delimiter=",")
            return DataSet, dataname
            break
        except FileNotFoundError:
            try:
                DataSet, dataname = np.loadtxt('regressiondata/{}.txt'.format(datarun), delimiter=","), datarun
                return DataSet, dataname
                break
            except FileNotFoundError:
                try:
                    DataSet, dataname = np.loadtxt('regressiondata/{}.csv'.format(datarun), delimiter=","), datarun
                    return DataSet, dataname
                    break
                except FileNotFoundError:
                    print('Dataset Not Found')

DataSet, dataname = CheckDataName()

cols = DataSet.shape[1]
X = DataSet[:, 0:cols - 1]
y = DataSet[:, cols - 1:cols]

lengthdata = len(X)
n_features = len(X[0])

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
# nntrain_y = train_y
# train_y = np.ravel(train_y)

if not os.path.exists("./regressionmodelparameters/{}".format(dataname)): os.makedirs("./regressionmodelparameters/{}".format(dataname))
file = open("regressionmodelparameters/{}/nclassesfeatures.txt".format(dataname), "w")
file.write(str(n_features) + "," + str(lengthdata))
file.flush()

print('\nData Length:', lengthdata)
print('Number of Features:', n_features)
print('Max y:', int(max(y)), '\nMin y:', int(min(y)))

ModelSelect()
RepeatedModelSelect()

