import pickle
import pandas as pd
from scipy.io import loadmat

NNpredict = pd.read_pickle('Data/NNpickle.pickle')
data = loadmat('Data/ex3data1.mat')

X = data['X']
y = data['y']

correct = [1 if a == b else 0 for (a, b) in zip(NNpredict, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))