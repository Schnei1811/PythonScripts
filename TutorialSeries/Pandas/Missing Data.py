'''
We have a few options when considering the existence of missing data.
Ignore it - Just leave it there
Delete it - Remove all cases. Remove from data entirely. This means forfeiting the entire row of data.
Fill forward or backwards - This means taking the prior or following value and just filling it in.
Replace it with something static - For example, replacing all NaN data with -9999.
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
    df.rename(columns={'Value': 'United States'}, inplace=True)
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

#grab_initial_state_data3()

fig = plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')
HPI_data['TX1yr'] = HPI_data['TX'].resample('A').mean()
#print(HPI_data[['TX','TX1yr']])

#HPI_data.dropna(inplace=True)
#HPI_data.dropna(how='all',inplace=True)        If all NAN
#HPI_data.fillna(method='ffill',inplace=True)       #Forward fill. Fills empty Data with the Data before
HPI_data.fillna(method='bfill',inplace=True)        #Backward fill. Fills empty Data with the Data after
#HPI_data.fillna(value=-99999,inplace=True)         #Fill NA with -99999 values
print(HPI_data[['TX','TX1yr']])

HPI_data['TX'].plot(ax=ax1)
HPI_data['TX1yr'].plot(color='k',ax=ax1)

plt.legend().remove()
plt.show()