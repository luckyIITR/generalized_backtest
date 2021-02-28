import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from Portfolio2 import Combine, Store_Data


def get_intra_data(symbol):
    daily = yf.download(tickers=symbol, interval="5m", period="20d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_dates(symbol):
    daily = yf.download(tickers=symbol, interval="1d", period="20d")
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


def get_today(df, symbol, date):
    return df[df.index.date == date]


def main(symbol):
    df = get_intra_data(symbol)
    dates = list(get_dates(symbol).index.date)
    port = Combine(symbol)
    for date in dates:
        today = get_today(df, symbol, date)
        for e in today.index:
            if e == today.index[0]: continue
            y1, y2, peaks1, peaks2 = get_peaks(today.loc[:e, ])
            if len(peaks1) == 0 or len(peaks2) == 0: continue
            if today.loc[e, 'High'] > y1[peaks1[-1]] and (port.check_pos() == 0 or port.check_pos() == -1):
                port.buy(y1[peaks1[-1]], e)

            elif today.loc[e, 'Low'] < y2[peaks2[-1]] * -1 and (port.check_pos() == 1):
                port.sell(y2[peaks2[-1]] * -1, e)

    port.generate_dataframes()
    store_result.append_data(port.generate_results())
    store_result.day_wise_result(port.get_day_wise())

store_result = Store_Data()

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
tickers = tickers[:10]
for symbol in tickers:
    main(symbol)

result, day_wise = store_result.gen_pd()  ## dont run it twice
store_result.get_csv()
