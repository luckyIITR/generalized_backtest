import Portfolio
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr


stocks = ["SBIN.NS"]
for stock in stocks:
    # stock=input("Enter a stock ticker symbol: ")
    print(stock)
    startyear = 2019
    endyear = 2020
    startmonth = 1
    startday = 1

    start = dt.datetime(startyear, startmonth, startday)

    now = dt.datetime(endyear, startmonth, startday)

    df = pdr.get_data_yahoo(stock, start, now)

    # ma=50

    # smaString="Sma_"+str(ma)

    # df[smaString]=df.iloc[:,4].rolling(window=ma).mean()

    emasUsed = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    for x in emasUsed:
        ema = x
        df["Ema_" + str(ema)] = round(df.iloc[:, 4].ewm(span=ema, adjust=False).mean(), 2)

    # df=df.iloc[60:]

    pos = 0
    num = 0
    percentchange = []

    for i in df.index:
        cmin = min(df["Ema_3"][i], df["Ema_5"][i], df["Ema_8"][i], df["Ema_10"][i], df["Ema_12"][i], df["Ema_15"][i], )
        cmax = max(df["Ema_30"][i], df["Ema_35"][i], df["Ema_40"][i], df["Ema_45"][i], df["Ema_50"][i],
                   df["Ema_60"][i], )

        close = df["Adj Close"][i]

        if (cmin > cmax):
            # print("Red White Blue")
            if (pos == 0):
                bp = close
                pos = 1
            # print("Buying now at "+str(bp))


        elif (cmin < cmax):
            # print("Blue White Red")
            if (pos == 1):
                pos = 0
                sp = close
                # print("Selling now at "+str(sp))
                pc = (sp / bp - 1) * 100
                percentchange.append(pc)
        if (num == df["Adj Close"].count() - 1 and pos == 1):
            pos = 0
            sp = close
            # print("Selling now at "+str(sp))
            pc = (sp / bp - 1) * 100
            percentchange.append(pc)

        num += 1

        '''[2.211636657021976,
 3.0113975429978623,
 -2.400833000354763,
 0.22463724814858477,
 5.268076377752107,
 3.8051142336279042,
 -0.11729427301431228,
 1.2953368387701447,
 14.77033007726629,
 0.8368828956117014,
 2.196495705119572,
 -10.657413981852649,
 1.4754702729415037,
 -15.32980224974323,
 20.97142392113096,
 2.1435209777305175,
 2.8932284730575963]'''