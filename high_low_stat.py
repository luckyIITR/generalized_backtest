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

plt.ioff()


#
# dbd = r'F:\Database\1min_data'
# db = sqlite3.connect(os.path.join(dbd, "NSEEQ.db"))
#
#
# def get_intra_data(symbol):
#     symbol_check = {'3MINDIA': 'MINDIA',
#                     'BAJAJ-AUTO': 'BAJAJAUTO',
#                     'J&KBANK': 'JKBANK',
#                     'L&TFH': 'LTFH',
#                     'M&MFIN': 'MMFIN',
#                     'M&M': 'MM',
#                     'NAM-INDIA': 'NAMINDIA',
#                     'MCDOWELL-N': 'MCDOWELLN'}
#     symbol = symbol[:-3]
#     if symbol in list(symbol_check.keys()):
#         symbol = symbol_check[symbol]
#
#     df = pd.read_sql('''SELECT * FROM %s;''' % symbol, con=db)
#     df.set_index('time', inplace=True)
#     df.reset_index(inplace=True)
#     df['time'] = pd.to_datetime(df['time'])
#     df.set_index("time", drop=True, inplace=True)
#     df.index[0]
#     df.drop(["oi", 'Volume'], axis=1, inplace=True)
#     return df


def get_intra_data(symbol, tp):
    daily = yf.download(tickers=symbol, interval="5m", period=f"{str(tp)}d")
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_dates(symbol, tp):
    daily = yf.download(tickers=symbol, interval="60m", period=f"{str(tp)}d")
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


def main(symbol):
    # symbol = "SBIN.NS"
    backtest_tp = 60
    df_5min = get_intra_data(symbol, backtest_tp)
    df_hour = get_dates(symbol, backtest_tp)
    df_hour['ema21'] = TA.EMA(df_hour, period=21)
    df_hour['ema8'] = TA.EMA(df_hour, period=8)
    df_hour = df_hour.iloc[21:, :].copy()
    df_hour['signal'] = [
        1 if df_hour.loc[e, 'ema8'] - df_hour.loc[e, 'ema21'] > 0 and df_hour.loc[e, 'Close'] > df_hour.loc[
            e, 'ema8'] else 0 for e in df_hour.index]
    df_5min = pd.concat([df_5min, df_hour['signal']], axis=1)
    df_5min = df_5min.ffill()
    df_5min.dropna(inplace=True)
    port = BuyPortfolio(symbol)

    dates = sorted(list(set(df_hour.index.date)))
    for date in dates:
        today = get_today(df_5min, date)
        for e in today.index:
            if e == today.index[0]:
                continue
            y1, y2, peaks1, peaks2 = get_peaks(today.loc[:e, ])
            if len(peaks1) == 0 or len(peaks2) == 0:
                continue

            if today.loc[e, 'signal'] == 1 and today.loc[e, 'High'] > y1[peaks1[-1]] and port.check_pos() == 0:
                port.buy(y1[peaks1[-1]], e)

            elif today.loc[e, 'Low'] < y2[peaks2[-1]] * -1 and port.check_pos() == 1:
                port.square_off(y2[peaks2[-1]] * -1, e)

            if port.check_pos() == 1 and e.time() == dt.datetime(2020, 2, 2, 15, 25).time():
                port.square_off(today.loc[e, 'Open'], e)
    port.generate_dataframes()
    store_result.append_data(port.generate_results())
    store_result.day_wise_result(port.get_day_wise().rename(columns={'%change': f"{symbol[:-3]}"}))

    # port.generate_csv_report()
    # fig = plt.figure(num=None, figsize=(16, 12), dpi=160, facecolor='w', edgecolor='k')
    # plt.plot(port.percent_df['cumprod'], 'bo-')
    # plt.xticks(rotation=45)
    # plt.xlabel('Date-time', fontsize=18)
    # plt.ylabel('Cumulative % change', fontsize=16)
    # plt.savefig(f"./plot/{symbol[:-3]}.jpeg")
    # plt.close(fig)


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
# tickers = tickers[:10]
for symbol in tickers:
    main(symbol)

result, day_wise = store_result.gen_pd(len(tickers))
store_result.get_csv()

# store_result.day_wise

# t = ((day_wise / 100 + 1).cumprod() - 1) * 100
