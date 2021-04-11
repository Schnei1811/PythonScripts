#http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

import Quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = 'EbuNm86FNrRBxTfzzynf'

def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]

def grab_initial_state_data3():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        df.columns = [str(abbv)]
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.rename(columns={'Value': 'United States'}, inplace=True)
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

#grab_initial_state_data3()

fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')

TX1yr = HPI_data['TX'].resample('A').mean()
#TX1yr = HPI_data['TX'].resample('A').ohlc()
print(TX1yr.head())

HPI_data['TX'].plot(ax=ax1)
TX1yr.plot(ax=ax1)

plt.legend().remove()
plt.show()

HPI_data.plot()
plt.legend().remove()
plt.show()



'''
Correlation is the measure of the degree by which two assets move in relation to each other.
Covariance is the measure of how two assets tend to vary together. Notice that correlation is
a measure to the "degree" of. Covariance isn't. That's the important distinction if my own
understanding is not incorrect!
'''

#HPI_State_Correlation = HPI_data.corr()
#print(HPI_State_Correlation)
#print(HPI_State_Correlation.describe())

'''
Resample rule:
xL for milliseconds
xMin for minutes
xD for Days

Alias	Description
B	business day frequency
C	custom business day frequency (experimental)
D	calendar day frequency
W	weekly frequency
M	month end frequency
BM	business month end frequency
CBM	custom business month end frequency
MS	month start frequency
BMS	business month start frequency
CBMS	custom business month start frequency
Q	quarter end frequency
BQ	business quarter endfrequency
QS	quarter start frequency
BQS	business quarter start frequency
A	year end frequency
BA	business year end frequency
AS	year start frequency
BAS	business year start frequency
BH	business hour frequency
H	hourly frequency
T	minutely frequency
S	secondly frequency
L	milliseonds
U	microseconds
N	nanoseconds

How:
mean, sum, ohlc
'''



'''
def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        df.columns = [str(abbv)]        #df.rename(columns = {'Value':str(abbv)}, inplace = True)﻿
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()
    print(main_df)

grab_initial_state_data()

pickle_in = open('fiddy_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)

HPI_data.to_pickle('pickle.pickle')
HPI_data2 = pd.read_pickle('pickle.pickle')
print(HPI_data2)


#HPI_data['TX2'] = HPI_data['TX'] * 2
#print(HPI_data[['TX','TX2']].head())

HPI_data.plot()
plt.legend().remove()
plt.show()


def grab_initial_state_data2():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        df.columns = [str(abbv)]
        print(query)
        df = df.pct_change()
        print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states2.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()


grab_initial_state_data2()


HPI_data = pd.read_pickle('fiddy_states2.pickle')

HPI_data.plot()
plt.legend().remove()
plt.show()
'''