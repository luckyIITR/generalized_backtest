from Portfolio import Combine
import datetime as dt
from pandas_datareader import data as pdr

stocks = ["SBIN.NS"]
for stock in stocks:
    port = Combine(stock)
    # stock=input("Enter a stock ticker symbol: ")
    print(stock)
    startyear = 2018
    endyear = 2021
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
    num = 0
    # pos = 0
    for i in df.index:
        cmin = min(df["Ema_3"][i], df["Ema_5"][i], df["Ema_8"][i], df["Ema_10"][i], df["Ema_12"][i], df["Ema_15"][i], )
        cmax = max(df["Ema_30"][i], df["Ema_35"][i], df["Ema_40"][i], df["Ema_45"][i], df["Ema_50"][i],
                   df["Ema_60"][i], )
        if cmin > cmax:
            # print("Red White Blue")
            if port.check_pos() == -1 or port.check_pos() == 0:

                # break
                port.buy(df["Adj Close"][i],i)
                # print("Buying now at "+str(bp))

        elif cmin < cmax:
            # print("Blue White Red")
            if port.check_pos() == 1:

                # break
                port.sell(df["Adj Close"][i],i)
                # print("Selling now at "+str(sp))

        # if num == df["Adj Close"].count() - 1 and port.check_pos() == -1:
        #     sp = close
        #     port.square_off(sp, time=i)
        # num += 1

