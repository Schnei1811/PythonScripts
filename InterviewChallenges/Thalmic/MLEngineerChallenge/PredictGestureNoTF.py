import numpy as np
import pandas as pd
import fnmatch
import os

TotalNumTestExamples = len(fnmatch.filter(os.listdir('TestData'), '*.txt'))
logitclf = pd.read_pickle('modelparameters/LOGITpickle.pickle')

def LoadGestureData(filename):
    InputLine = []
    df = np.loadtxt(filename, delimiter=",")
    for i, j in enumerate(df):
        X = df[i, :]
        InputLine = np.insert(X, 0, InputLine)
    return InputLine

'''
Please place Testing Examples in the TestData folder. There are included examples to help answer any questions.

If you plan on loading invidiual files, modify the file name here
'''

filename = 'TestData/Gesture2Test.txt'
print('Predicted Gesture:', int(logitclf.predict(np.array([LoadGestureData(filename)]))))

'''
If you'd like to iteratively load example files from the folder, modify the file name here
'''

for i in range(0, TotalNumTestExamples):
    filename = 'TestData/Gesture{}Test.txt'.format(i+1)
    print('Predicted Gesture:', int(logitclf.predict(np.array([LoadGestureData(filename)]))))


'''
If you have any questions, comments, or concerns please let me know
'''