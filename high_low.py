from scipy.signal import argrelextrema
import numpy as np
import yfinance as yf
import datetime as dt
import pandas as pd
from finta import TA
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def get_intra_data(symbol):
    daily = yf.download(tickers=symbol,  interval="5m", period = "50d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    # daily = daily.ffill()
    # daily['sig'] = daily['minema']- daily['hourema']
    return daily

def get_dates(symbol):
    daily = yf.download(tickers=symbol, interval="1d", period = "50d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily

def get_peaks(today):
    y1 = today.loc[:, 'High'].values
    x = np.array([i for i in range(1, len(y1) + 1)])
    peaks1, _ = find_peaks(y1)


    y2 = today.loc[:, 'Low'].values * -1
    x = np.array([i for i in range(1, len(y1) + 1)])
    peaks2, _ = find_peaks(y2)
    return y1,y2,peaks1,peaks2

def get_plot(y1,y2,peaks1,peaks2,df):
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

def get_today(df,symbol,date):
    return df[df.index.date == date]

def daywise(ti,percen):
    p_l = []
    buypl = pd.DataFrame()
    buypl = pd.DataFrame.from_dict({"time":ti,"%":percen}, orient='columns', dtype=None, columns=None)
    buypl.set_index("time", drop=True, inplace=True)
    dates = list(set(buypl.index.date))
    dates = sorted(dates)
    for date in dates:
        p_l.append(buypl[buypl.index.date == date].sum()[0])
    buypl = pd.DataFrame()
    buypl = pd.DataFrame.from_dict({"time":dates,"%":p_l}, orient='columns', dtype=None, columns=None)
    buypl.set_index("time", drop=True, inplace=True)
    return buypl

def main(symbol) :
    ti = []
    percen = []
    df = get_intra_data(symbol)
    dates = list(get_dates(symbol).index.date)
    for date in dates:
        # print("########## new day")
        today = get_today(df,symbol,date)
        y1,y2,peaks1,peaks2 = get_peaks(today)
        pos = 0
        i = 0
        flag = 0
        h1 =0
        l1 =0
        # percen = []
        sl = 99999
        for e in today.index:
            # searching for signal
            if i in peaks1 :
                h1 = y1[i]
            if i in peaks2 :
                l1 = y2[i]*-1
            if (h1 == 0 or l1 == 0):
                i = i + 1
                # flag = 1
                continue

            if pos == 0 and today.loc[e,'High'] > h1:
                bp = h1
                sl = l1
                pos = 1
                # print(f"buyed at {bp} time {e} and sl at {sl}")
                i = i + 1
                continue
            if pos == 1 and today.loc[e,'Low'] <= sl:
                sp = sl
                per = (sp/bp-1)*100
                # print(f"TSL hit {sp} time {e}")
                percen.append(per)
                ti.append(e)
                pos = 0
            if l1 > sl and pos == 1:
                sl = l1
                # print(f"sl trailed to {sl}")
            if pos == 1 and e.time() == dt.datetime(2020,2,2,15,25) :
                sp = today.loc[e,'Open']
                per = (sp / bp - 1) * 100
                # print(f"Sold at {sp}")
                percen.append(per)
                ti.append(e)
                pos = 0
            i = i + 1
        # print(sum(percen))

    pl_buy = daywise(ti,percen)

    percen = []
    ti = []
    df = get_intra_data(symbol)
    dates = list(get_dates(symbol).index.date)
    for date in dates:
        # print("########## new day")
        today = get_today(df,symbol,date)
        y1,y2,peaks1,peaks2 = get_peaks(today)
        pos = 0
        i = 0
        flag = 0
        h1 =0
        l1 =0
        # percen = []
        sl = 0
        for e in today.index:
            # searching for signal
            if i in peaks1 :
                h1 = y1[i]
            if i in peaks2 :
                l1 = y2[i]*-1
            if (h1 == 0 or l1 == 0):
                i = i + 1
                # flag = 1
                continue

            if pos == 0 and today.loc[e,'Low'] < l1:
                sp = l1
                sl = h1
                pos = 1
                # print(f"selling at {sp} time {e} and sl at {sl}")
                i = i + 1
                continue
            if pos == 1 and today.loc[e,'High'] >= sl:
                bp = sl
                per = (sp/bp-1)*100
                # print(f"TSL hit {bp} time {e}")
                ti.append(e)
                percen.append(per)
                pos = 0
            if h1 < sl and pos == 1:
                sl = h1
                # print(f"sl trailed to {sl}")
            if pos == 1 and e.time() == dt.datetime(2020,2,2,15,25) :
                bp = today.loc[e,'Open']
                per = (sp / bp - 1) * 100
                # print(f"brought at {bp}")
                percen.append(per)
                ti.append(e)
                pos = 0
            i = i + 1
        # print(sum(percen))

    pl_sell = daywise(ti,percen)

    pl_buy['sell'] = pl_sell
    pl_buy.fillna(0,inplace=True)
    pl_buy['net'] = pl_buy["%"] + pl_buy['sell']

    x = ((pl_buy['net']/100+1).cumprod() - 1)*100
    plt.plot(x)
    plt.show()
    print(f"{symbol} : {x[-1]}")
    return x[-1]
tickers = ['ADANIPORTS.NS',
     'ASIANPAINT.NS',
     'AXISBANK.NS',
     'BAJAJ-AUTO.NS',
     'BAJAJFINSV.NS',
     'BAJFINANCE.NS',
     'BHARTIARTL.NS',
     'BPCL.NS',
     'BRITANNIA.NS',
     'CIPLA.NS',
     'COALINDIA.NS',
     'DIVISLAB.NS',
     'DRREDDY.NS',
     'EICHERMOT.NS',
     'GAIL.NS',
     'GRASIM.NS',
     'HCLTECH.NS',
     'HDFC.NS',
     'HDFCBANK.NS',
     'HDFCLIFE.NS',
     'HEROMOTOCO.NS',
     'HINDALCO.NS',
     'HINDUNILVR.NS',
     'ICICIBANK.NS',
     'INDUSINDBK.NS',
     'INFY.NS',
     'IOC.NS',
     'ITC.NS',
     'JSWSTEEL.NS',
     'KOTAKBANK.NS',
     'LT.NS',
     'M&M.NS',
     'MARUTI.NS',
     'NESTLEIND.NS',
     'NTPC.NS',
     'ONGC.NS',
     'POWERGRID.NS',
     'RELIANCE.NS',
     'SBILIFE.NS',
     'SBIN.NS',
     'SUNPHARMA.NS',
     'TATAMOTORS.NS',
     'TATASTEEL.NS',
     'TCS.NS',
     'TECHM.NS',
     'TITAN.NS',
     'ULTRACEMCO.NS',
     'UPL.NS',
     'WIPRO.NS']

symbol_return = []
for symbol in tickers:
    # symbol = "HDFCBANK.NS"
    symbol_return.append(main(symbol))

p_l = []
buypl = pd.DataFrame()
buypl = pd.DataFrame.from_dict({"symbol":tickers,"%":symbol_return}, orient='columns', dtype=None, columns=None)
buypl.to_csv("alert_data.csv")
# get_plot(y1,y2,peaks1,peaks2,today)