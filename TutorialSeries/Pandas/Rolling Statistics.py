#http://pandas.pydata.org/pandas-docs/version/0.15.2/computation.html#moving-rolling-statistics-moments
'''
One of the more popular rolling statistics is the moving average. This takes a moving window of time,
and calculates the average or the mean of that time period as the current value. In our case, we have monthly Data.
'''

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
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

fig = plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
HPI_data = pd.read_pickle('fiddy_states3.pickle')
#HPI_data['TX12MA'] = pd.rolling_mean(HPI_data['TX'], 12)
#HPI_data['TX12STD'] = pd.rolling_std(HPI_data['TX'], 12)

#HPI_data['TX'].plot(ax=ax1)
#HPI_data['TX12MA'].plot(ax=ax1)
#HPI_data['TX12STD'].plot(ax=ax2)

#plt.show()

TX_AK_12corr = pd.rolling_corr(HPI_data['TX'], HPI_data['AK'], 12)

HPI_data['TX'].plot(ax=ax1, label="TX HPI")
HPI_data['AK'].plot(ax=ax1, label="AK HPI")
ax1.legend(loc=4)

TX_AK_12corr.plot(ax=ax2)

plt.show()