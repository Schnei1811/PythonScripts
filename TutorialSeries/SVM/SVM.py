#Binary Classifier but can used for multiple classification problems
#Separates 1 group in comparison to the others
#Solve for best separable hyperplane. Distance between hyperplane and Data is the greatest
#Hyperplane through classification Data. Xi . W + b = 1 for + = -1 for -. Known as Support Vectors
#Solving for the hyperplane decision boundary
#Optimization objective: Minimize ||W|| and Maximize b (bias)
#Convex problem

#Multiclassification SVM - Binary classifier. Can only separate one group at a time
#OVR - One Versus Rest
#Common to have uneven numbered Data points. Generally the default used
#OVO - One Versus One
#1 vs 2 1 vs 3  2 v 3
# decision_fucntion_shape: ovo, ovr or None
# can grab w and b to visualize dataset

#Google SVM.svc

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve

def ROCcalculation(X, y, predict):
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
        if (y[i] == (1) and (predict(x) == (1))):
            tpcount = tpcount + 1
            tprate[i] = tpcount / sum(y_test - 1)
            tp = tprate[i]
            fprate[i] = fp
        elif (y[i] == (1) and (predict(x) == (2))):
            fncount = fncount + 1
            tprate[i] = tp
            fprate[i] = fp
        elif (y[i] == (2) and (predict(x) == (1))):
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

df = pd.read_csv('KNNcancerdata')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[1,1,1,1,1,1,1,1,1], [3,2,5,1,2,7,6,5,4]])

example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(svm.SVC(), X, y, param_name="gamma", param_range=param_range,cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

tp, fp, fn, tn, F1, fpratetest, tpratetest = ROCcalculation(X_test,y_test, clf.predict)
print(tp,fp,fn, tn)

plt.plot(fpratetest, tpratetest, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Match Overview - Simple Neural Network ROC')
plt.legend(loc=4)
plt.show()

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.show()

print("done")