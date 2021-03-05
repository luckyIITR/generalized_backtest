import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from Portfolio2 import BuyPortfolio, Store_Data
# import os
# import sqlite3
import pandas as pd
import datetime as dt
from finta import TA
from plotly.offline import plot
from scipy import interpolate

def get_data(symbol, interval, tp):
    daily = yf.download(tickers=symbol, interval=f"{str(interval)}m", period=f"{str(tp)}d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_peaks(macd):
    y1 = macd.loc[:, 'SIGNAL'].values
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks1, _ = find_peaks(y1)

    y2 = macd.loc[:, 'SIGNAL'].values * -1
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks2, _ = find_peaks(y2)
    return y1, y2*-1, peaks1, peaks2


def get_plot(df):
    trace2 = {
        'x': df.index,
        'y': df.MACD
    }
    trace1 = {
        'x': df.index,
        'y': df.SIGNAL,
    }
    data = [trace1,trace2]
    fig = dict(data=data)
    plot(fig)


def get_today(df, date):
    return df[df.index.date == date]

def get_sig_data(symbol):
    df_5min = get_data(symbol, 5, 60)
    df_15min = get_data(symbol, 15, 60)
    df_hour = get_data(symbol, 60, 60)
    macd = TA.MACD(df_5min)
    df_15min['mtfema15'] = TA.EMA(df_15min, 50).shift(1)
    df_hour['mtfema50'] = TA.EMA(df_hour, 50).shift(1)
    df_hour = df_hour.iloc[50:, :].copy()
    df_5min = df_5min.loc[df_hour.index[0]:, :]
    df_15min = df_15min.loc[df_hour.index[0]:, :]
    macd = macd.loc[df_hour.index[0]:, :]
    df_5min = pd.concat([df_5min,macd],axis = 1)
    df_5min = pd.concat([df_5min, df_15min['mtfema15'], df_hour['mtfema50']], axis=1)
    df_5min = df_5min.ffill()
    df_5min['signal'] = [1 if df_5min.loc[e, 'mtfema15'] > df_5min.loc[e, 'mtfema50'] else 0 for e in df_5min.index]
    return df_5min


def get_today(df, date):
    return df[df.index.date == date]


def main():
    symbol = "SBIN.NS"
    df_5min = get_sig_data(symbol)


    port = BuyPortfolio(symbol)
    dates = sorted(list(set(df_5min.index.date)))

    date = dates[-2]
    day = []
    for date in dates:
        today = get_today(df_5min, date)
        y1, y2, peaks1, peaks2 = get_peaks(today)
        if len(y2[peaks2]) >1:
            day.append(date)
            print(f"date : {date}  {y2[peaks2]}")



        plt.plot(df_5min[["MACD", "SIGNAL"]][df_5min.index.date ==date])
        plt.plot(df_5min[["MACD", "SIGNAL"]][df_5min.index.date ==date].index[peaks1],y1[peaks1],'x')
        plt.plot(df_5min[["MACD", "SIGNAL"]][df_5min.index.date ==date].index[peaks2],y1[peaks2],'x')
