import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import sys
import pickle
from sklearn import preprocessing, cross_validation,svm, neighbors,tree
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

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
    i=+1

X_train, X_test, y_train, y_test = cross_validation.train_test_split(xdata,ydata,test_size=0.19365,random_state=1)

input_size = 226
hidden_size = 1000
num_labels = 2
learning_rate = 1
maxiter = 10

X = X_train
y = y_train

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

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
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####

    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    return J

def ROCcalculation(X, y):
    tpcount = 0
    fpcount = 0
    tncount = 0
    fncount = 0
    tprate = np.zeros((y.shape[0], 1))
    fprate = np.zeros((y.shape[0], 1))
    tp = 0
    fp = 0
    tn = X.shape[0] - sum(y - 1)

    for i in range(len(X_test)):
        x = np.array([X_test[i]])
        a1, z2, a2, z3, h = forward_propagate(x, theta1, theta2)
        pred = np.array(np.argmax(h, axis=1) + 1)
        if (y[i] == (1) and (pred == 1)):
            tpcount = tpcount + 1
            tprate[i] = tpcount / sum(y - 1)
            tp = tprate[i]
            fprate[i] = fp
        elif (y[i] == (1) and (pred == 2)):
            fncount = fncount + 1
            tprate[i] = tp
            fprate[i] = fp
        elif (y[i] == (2) and (pred == (1))):
            fpcount = fpcount + 1
            fprate[i] = fpcount / tn
            tprate[i] = tp
            fp = fprate[i]
        else:
            tncount = tncount + 1
            tprate[i] = tp
            fprate[i] = fp

    precision = tpcount / (tpcount + fpcount)
    recall = tpcount / (tpcount + fncount)
    TestF1 = (2 * precision * recall) / (precision + recall)

    return tpcount, fpcount, fncount, tncount, TestF1, fprate, tprate

def ROCcalculation2(X, y, model):
    tpcount = 0
    fpcount = 0
    tncount = 0
    fncount = 0
    tprate = np.zeros((y.shape[0], 1))
    fprate = np.zeros((y.shape[0], 1))
    tp = 0
    fp = 0
    tn = X.shape[0] - sum(y - 1)

    for i in range(len(X)):
        x = np.array([X[i]])
        if (y[i] == (1) and (model.predict(x) == (1))):
            tpcount = tpcount + 1
            tprate[i] = tpcount / sum(y - 1)
            tp = tprate[i]
            fprate[i] = fp
        elif (y[i] == (1) and (model.predict(x) == (2))):
            fncount = fncount + 1
            tprate[i] = tp
            fprate[i] = fp
        elif (y[i] == (2) and (model.predict(x) == (1))):
            fpcount = fpcount + 1
            fprate[i] = fpcount / tn
            tprate[i] = tp
            fp = fprate[i]
        else:
            tncount = tncount + 1
            tprate[i] = tp
            fprate[i] = fp

    precision = tpcount / (tpcount + fpcount)
    recall = tpcount / (tpcount + fncount)
    TestF1 = (2 * precision * recall) / (precision + recall)

    return tpcount, fpcount, fncount, tncount, TestF1, fprate, tprate


params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

X = np.matrix(X)
y = np.matrix(y)

theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
print(J, grad.shape)

fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': maxiter})

X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))

a1, z2, a2, z3, h = forward_propagate(X_test, theta1, theta2)
y_predtest = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_predtest, y_test)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))

tp, fp, fn, tn, F1, fprateNN, tprateNN = ROCcalculation(X_test,y_test)
print(tp,fp,fn,tn)
print(F1)

#knearclf = neighbors.KNeighborsClassifier(n_jobs=-1)
#rbfsvmclf = svm.SVC(C=1,kernel='rbf',max_iter=100000)
#treeclf = tree.DecisionTreeClassifier()

#knearscore = knearclf.fit(X_train, y_train)
#rbfsvmscore = rbfsvmclf.fit(X_train, y_train)
#treescore = treeclf.fit(X_train,y_train)

#print(accuracy_score(y_test,knearclf.predict(X_test)))
#print(accuracy_score(y_test,rbfsvmclf.predict(X_test)))
#print(accuracy_score(y_test,treeclf.predict(X_test)))

#tp, fp, fn, tn, F1, fprateknn, tprateknn = ROCcalculation2(X_test,y_test, knearclf)
#print(tp,fp,fn,tn)
#print(F1)

#tp, fp, fn, tn, F1, fpraterbfsvm, tpraterbfsvm = ROCcalculation2(X_test,y_test, rbfsvmclf)
#print(tp,fp,fn,tn)
#print(F1)

#tp, fp, fn, tn, F1, fpratetree, tpratetree = ROCcalculation2(X_test,y_test, treeclf)
#print(tp,fp,fn,tn)
#print(F1)

#plt.plot(fprateNN, tprateNN, label = 'NN ROC Curve')
#plt.plot(fprateknn, tprateknn, label = 'KNN ROC Curve')
#plt.plot(fpraterbfsvm, tpraterbfsvm, label = 'SVM ROC Curve')
#plt.plot(fpratetree, tpratetree, label = 'Tree ROC Curve')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Match Overview - ROC')
#plt.axis([0,1,0,1])
#plt.legend(loc=4)
#plt.show()

savepickle = open('DecoyDota2MatchOverviewPickle.pickle','wb')
pickle.dump(fmin,savepickle)