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


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = Quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)
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
    df["United States"] = (df["United States"] - df["United States"][0]) / df["United States"][0] * 100.0
    df.rename(columns={'United States': 'US_HPI'}, inplace=True)
    return df

def mortgage_30y():
    df = Quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"] - df["Value"][0]) / df["Value"][0] * 100.0
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df

def sp500_data():
    df = Quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
    df["Adjusted Close"] = (df["Adjusted Close"] - df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={'Adjusted Close': 'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = Quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"] - df["Value"][0]) / df["Value"][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={'Value': 'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = Quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
    df["Unemployment Rate"] = (df["Unemployment Rate"] - df["Unemployment Rate"][0]) / df["Unemployment Rate"][
        0] * 100.0
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df

grab_initial_state_data()
HPI_data = pd.read_pickle('fiddy_states3.pickle')
m30 = mortgage_30y()
sp500 = sp500_data()
gdp = gdp_data()
HPI_Bench = HPI_Benchmark()
unemployment = us_unemployment()
m30.columns = ['M30']
HPI = HPI_Bench.join([m30, sp500, gdp, unemployment])
HPI.dropna(inplace=True)
print(HPI.corr())

#HPI_data = pd.read_pickle('fiddy_states3.pickle')
#HPI_bench = HPI_Benchmark()

#state_HPI_M30 = HPI_data.join(m30)

#print(state_HPI_M30.corr())
#print(state_HPI_M30.corr()['M30'].describe())      #Correlation of mortgages with state housing rates
#print(HPI_data.head())

#People trade contract that are indiciative of what people believe interest rates will be. Can extrapolate
# what people believe in the now

