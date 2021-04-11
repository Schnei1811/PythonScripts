import numpy as np

class GDC:
    NumExample = 1

    def __init__(self, TotalNumExamples, TotalPoorDataCount, NumGesture):
        self.TotalNumExamples = TotalNumExamples
        self.TotalPoorDataCount = TotalPoorDataCount
        self.NumGesture = NumGesture

    def DataCreation(self):
        while GDC.NumExample <= self.TotalNumExamples:
            self.InputLine = []
            #df = np.loadtxt('Gesture Data/Gesture{}_Example{}.txt'.format(NumGesture, NumExample), delimiter=",")
            self.df = np.loadtxt('TestData/Gesture{}Test.txt'.format(self.NumGesture), delimiter=",")
            for i, j in enumerate(self.df):
                self.X = self.df[i, :]
                self.InputLine = np.insert(self.X, 0, self.InputLine)
            if GDC.NumExample == 1:
                self.GestureData = self.InputLine
            else:
                try:
                    self.GestureData = np.vstack((self.GestureData, self.InputLine))
                except ValueError:
                    self.TotalPoorDataCount += 1
                    print(print('Data Error. Gesture:', self.NumGesture, ' Example:', self.NumExample, ' Poor Data Count Number:', self.TotalPoorDataCount))
            GDC.NumExample += 1
        self.Classification = np.ones((len(self.GestureData),1))*self.NumGesture
        if GDC.NumExample > 1:
            self.GestureData = np.concatenate((self.GestureData, self.Classification), axis=1)
        return self.GestureData, self.TotalPoorDataCount
