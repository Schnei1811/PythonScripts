#Regression     Continuous data. Straight line through the data

import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

#dataframe
df = Quandl.get('WIKI/GOOGL')

#print(df.head())
#prints first five rows of the dataframes
#adjusted after things like stock splits. Split shares to lower share price (1 $1000 vs 2 $500)
#want as many meaningful features as you can get
#major high/low illustrates volatilty

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#define new coloumn
#High minus low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
#Percent Change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#   wouldn't use these features but features that predict a companies overall value
#   quarterly earnings, price to earnings, price to earnings to growth, book value, etc.
#           price      not price   not price     not price
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)   #Replace NAN Data with something. Treated as outlier later on
forecast_out = int(math.ceil(0.1 * len(df)))   #Rounds everything up to the nearest whole. Predict out 10% of the dataframe
                                            #Change to predict price of varying lengths of time
#print(forecast_out)     #days in advance
df['label'] = df[forecast_col].shift(-forecast_out)   #int for shifting coloumn
# shifting coloumn negatively. Label coloumn for each row will be the adjusted close for a 1% change
# label coloumn. Time into the future
#print(df.tail())
#print(df.head())

forecast_col = 'Adj. Close'

X = np.array(df.drop(['label','Adj. Close'], 1))  #Everything but the label drop
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]     #Includes Xs where we have values for y
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X),len(y))        #check lengths

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf = svm.SVR()
#clf = svm.SVR(kernel='poly')
#clf = LinearRegression(n_jobs = 10)    Will run in 10 threads in parallel
#clf = LinearRegression(n_jobs = -1)    Will run as many threads as capable by the processor
    # Understand how many can be used for a given algorithm

#Unneeded after pickle is trained
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
with open('linearregression.pickle','wb') as f:  #save classifier to avoid training step. Temp variable f
    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)     #accuracy is the squared error
#print(accuracy)
forecast_set = clf.predict(X_lately)    # can have a single or array of values. Each value one day away
print(forecast_set, accuracy, forecast_out)     #Next 30 days of unknown values

df['Forecast'] = np.nan         # coloumn all NAN values
last_date = df.iloc[-1].name    # date is not a feature. Need to create feature
last_unix = last_date.timestamp()
one_day = 86400                 # 86400 seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:          #taking each forecast and day, setting them as the values in the df. Making the future features not a number
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]    #Sets the first coloumn to a Nan and then i to the forecast
        # Referencing the index for the dataframe. Next date is a date stamp. Is the index of the dataframe
        # list of values are NAN plus i

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Pickle - Serialization of any python object. Could be dictionary, classifier, many things

# Look at documentation for algoirthm to look at jobs. How many jobs (threads) are we willing to run at any given time
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
