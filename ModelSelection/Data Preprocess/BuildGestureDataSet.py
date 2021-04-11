import numpy as np

def GestureDataCreation(TotalPoorDataCount, NumGesture):
    NumExample = 1
    while NumExample <= TotalNumExamples:
        InputLine = []
        df = np.loadtxt('Gesture Data/Gesture{}_Example{}.txt'.format(NumGesture, NumExample), delimiter=",")
        for i, j in enumerate(df):
            X = df[i, :]
            InputLine = np.insert(X, 0, InputLine)
        if NumExample == 1: GestureData = InputLine
        else:
            try: GestureData = np.vstack((GestureData,InputLine))
            except ValueError:
                TotalPoorDataCount += 1
                print('Data Error. Gesture:', NumGesture, ' Example:', NumExample, ' Poor Data Count Number:', TotalPoorDataCount)
                #Option1: Drop Data Entry
                #Option2: Insert Zeros for Missing Values
                #Option3: Use Imputation to Solve for Best Estimates of Missing Values
        NumExample += 1
    Classification = np.ones((len(GestureData),1))*NumGesture
    GestureData = np.concatenate((GestureData,Classification),axis=1)
    return GestureData, TotalPoorDataCount

TotalNumGestures = 6
TotalNumExamples = 2000
TotalPoorDataCount = 0

Gesture1Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=1)
Gesture2Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=2)
Gesture3Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=3)
Gesture4Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=4)
Gesture5Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=5)
Gesture6Data, TotalPoorDataCount = GestureDataCreation(TotalPoorDataCount, NumGesture=6)

DataSet = np.vstack((Gesture1Data,Gesture2Data))
DataSet = np.vstack((DataSet,Gesture3Data))
DataSet = np.vstack((DataSet,Gesture4Data))
DataSet = np.vstack((DataSet,Gesture5Data))
DataSet = np.vstack((DataSet,Gesture6Data))

np.savetxt('GestureData.txt', DataSet, fmt='%i',delimiter=',')