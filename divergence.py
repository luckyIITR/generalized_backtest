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


def get_data(symbol, interval, tp):
    daily = yf.download(tickers=symbol, interval=f"{str(interval)}m", period=f"{str(tp)}d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_peaks(today):
    y1 = today.loc[:, 'High'].values
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks1, _ = find_peaks(y1)

    y2 = today.loc[:, 'Low'].values * -1
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks2, _ = find_peaks(y2)
    return y1, y2, peaks1, peaks2


def get_plot(y1, y2, peaks1, peaks2, df):
    plt.plot(peaks1, y1[peaks1], "x")
    plt.plot(peaks2, y2[peaks2] * -1, "x")
    y = df.loc[:, 'Close'].values
    x = np.array([i for i in range(1, len(y1) + 1)])
    plt.plot(x - 1, y)
    for x1 in peaks1:
        plt.vlines(x=x1, ymin=min(y2 * -1), ymax=max(y1), colors='green', ls=':', lw=1)
    for x1 in peaks2:
        plt.vlines(x=x1, ymin=min(y2 * -1), ymax=max(y1), colors='green', ls=':', lw=1)

    for x1 in y1[peaks1]:
        plt.hlines(y=x1, xmin=min(x), xmax=max(x), colors='red', ls=':', lw=1)

    for x1 in y2[peaks2] * -1:
        plt.hlines(y=x1, xmin=min(x), xmax=max(x), colors='green', ls=':', lw=1)


def get_today(df, date):
    return df[df.index.date == date]



def main():
    symbol = "SBIN.NS"
    df_5min = get_data(symbol, 5,60)
    df_15min = get_data(symbol,15,60)
    df_hour = get_data(symbol,60,60)
    df_15min['mtfema15'] = TA.EMA(df_15min,50)
    df_hour['mtfema50'] = TA