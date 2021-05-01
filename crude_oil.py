import pandas as pd
# import os
# import sqlite3
from scipy.signal import find_peaks
from Portfolio2 import Combine
import datetime as dt
from plotly.offline import plot
import yfinance as yf
from finta import TA
import numpy as np
import matplotlib.pyplot as plt


def get_intra_data(symbol, interval):
    daily = yf.download(tickers=symbol, interval=interval, period='30d')
    # daily = yf.download(tickers=symbol, interval="5m", period=f"{str(tp)}d" )
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_today(df, dat):
    return df[df.index.date == dat]


def get_peaks(today):
    y1 = today.loc[:, 'High'].values
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks1, _ = find_peaks(y1)

    y2 = today.loc[:, 'Low'].values * -1
    # x = np.array([i for i in range(1, len(y1) + 1)])
    peaks2, _ = find_peaks(y2)
    return y1, y2, peaks1, peaks2


def get_plot(y1, y2, peaks1, peaks2, df):
    trace1 = {
        'x': df.index,
        'open': df.Open,
        'close': df.Close,
        'high': df.High,
        'low': df.Low,
        'type': 'candlestick',
        'name': 'Crude',
        'showlegend': True
    }
    trace2 = {
        'x': df.index[peaks1],
        'y': y1[peaks1],
        'type': 'scatter'
    }
    trace3 = {
        'x': df.index[peaks2],
        'y': y2[peaks2] * -1,
        'type': 'scatter'
    }
    data = [trace1, trace2, trace3]
    fig = dict(data=data)
    plot(fig)


df_5min = get_intra_data("CL=F", "5m")

df_hour = get_intra_data('CL=F', '60m')
df_hour['ema21'] = TA.EMA(df_hour, period=21)
df_hour['ema8'] = TA.EMA(df_hour, period=8)
df_hour = df_hour.iloc[21:, :].copy()

sig = []
for e in df_hour.index:
    if df_hour.loc[e, 'ema8'] - df_hour.loc[e, 'ema21'] > 0 and df_hour.loc[e, 'Close'] > df_hour.loc[e, 'ema8']:
        sig.append(1)
    elif df_hour.loc[e, 'ema8'] - df_hour.loc[e, 'ema21'] < 0 and df_hour.loc[e, 'Close'] < df_hour.loc[e, 'ema8']:
        sig.append(-1)
    else:
        sig.append(0)
df_hour['signal'] = sig
df_hour['signal'] = df_hour['signal'].shift(1)
df_5min = pd.concat([df_5min, df_hour['signal']], axis=1)
df_5min = df_5min.ffill()
df_5min.dropna(inplace=True)

# data.index[0]
dates = sorted(list(set(df_5min.index.date)))[1:-1]
port = Combine("crude")
for date in dates:
    # break
    today = get_today(df_5min, date)
    count = 0
    for e in today.index:
        if e == today.index[0]:
            continue
        y1, y2, peaks1, peaks2 = get_peaks(today.loc[:e, ])
        if len(peaks1) == 0 or len(peaks2) == 0:
            continue

        if today.loc[e, 'High'] > y1[peaks1[-1]] and (port.check_pos() == 0 or port.check_pos() == -1):
            if count % 2 == 0:
                if today.loc[e, 'signal'] == 1:
                    port.buy(y1[peaks1[-1]], e)
                    count += 1
            elif count % 2 != 0:
                port.buy(y1[peaks1[-1]], e)
                count += 1

        elif today.loc[e, 'Low'] < y2[peaks2[-1]] * -1 and (port.check_pos() == 1 or port.check_pos() == 0):
            if count % 2 == 0:
                if today.loc[e, 'signal'] == -1:
                    port.sell(y2[peaks2[-1]] * -1, e)
                    count += 1
            elif count % 2 != 0:
                port.sell(y2[peaks2[-1]] * -1, e)
                count += 1

        if port.check_pos() == 1 and e.time() == dt.datetime(2020, 2, 2, 23, 55).time() and count % 2 != 0:
            port.sell(today.loc[e, 'Open'], e)

        elif port.check_pos() == -1 and e.time() == dt.datetime(2020, 2, 2, 23, 55).time() and count % 2 != 0:
            port.buy(today.loc[e, 'Open'], e)

    # today = today.iloc[:-2,:]
    # y1, y2, peaks1, peaks2 = get_peaks(today)
    # get_plot(y1, y2, peaks1, peaks2, today)

port.generate_dataframes()
