import yfinance as yf
from scipy.signal import find_peaks
from Portfolio3 import Combine
import pandas as pd
import datetime as dt
from finta import TA
from plotly.offline import plot
import matplotlib.pyplot as plt


def get_data(symbol, interval, start, end):
    daily = yf.download(tickers=symbol, interval=interval, start=start, end=end)
    # daily = yf.download(tickers=symbol, interval="5m", period=f"{str(tp)}d" )
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
    trace1 = {
        'x': df.index,
        'open': df.Open,
        'close': df.Close,
        'high': df.High,
        'low': df.Low,
        'type': 'candlestick',
        'name': symbol[:-3],
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
    # plt.plot(df.index[peaks1], y1[peaks1], "x")
    # plt.plot(df.index[peaks2], y2[peaks2] * -1, "x")
    # y = df.loc[:, 'Close'].values
    # x = np.array([i for i in range(1, len(y1) + 1)])
    # plt.plot(df.index, y)
    # for x1 in df.index[peaks1]:
    #     plt.vlines(x=x1, ymin=min(y2 * -1), ymax=max(y1), colors='green', ls=':', lw=1)
    # for x1 in df.index[peaks2]:
    #     plt.vlines(x=x1, ymin=min(y2 * -1), ymax=max(y1), colors='green', ls=':', lw=1)
    #
    # for x1 in y1[peaks1]:
    #     plt.hlines(y=x1, xmin=min(df.index), xmax=max(df.index), colors='red', ls=':', lw=1)
    #
    # for x1 in y2[peaks2] * -1:
    #     plt.hlines(y=x1, xmin=min(df.index), xmax=max(df.index), colors='green', ls=':', lw=1)


def get_today(df, date):
    return df[df.index.date == date]


# define variables
symbol = "CL=F"
start_d = dt.datetime(2021, 2, 16)
end_d = dt.datetime(2021, 3, 13)

# get Data
df_5min = get_data(symbol, '15m', start_d, end_d)
df_hour = get_data(symbol, '60m', start_d, end_d)

# calculation
df_hour['ema21'] = TA.EMA(df_hour, period=21)
df_hour['ema8'] = TA.EMA(df_hour, period=8)
df_hour = df_hour.iloc[21:, :].copy()


# signal creation
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


# object creation
port = Combine(symbol)

# get dates
dates = sorted(list(set(df_5min.index.date)))[1:-2]

# Backtest looped
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
            port.sell(y2[peaks2[-1]] * -1, e)
        elif today.loc[e, 'signal'] == -1 and today.loc[e, 'Low'] < y2[peaks2[-1]] * -1 and port.check_pos() == 0:
            port.sell(y2[peaks2[-1]] * -1, e)
        elif today.loc[e, 'High'] > y1[peaks1[-1]] and port.check_pos() == -1:
            port.buy(y1[peaks1[-1]], e)

        if port.check_even() == False and e.time() == dt.datetime(2020, 2, 2, 23, 55).time():
            if port.check_pos() == 1:
                port.sell(today.loc[e, 'Open'], e)
            elif port.check_pos() == -1:
                port.buy(today.loc[e, 'Open'], e)

if len(port.order_book) != 0:
    port.generate_dataframes()
    port.generate_results()
    # store_result.append_data(port.generate_results())
    # store_result.day_wise_result(port.get_day_wise().rename(columns={'%change': f"{symbol}"}))

    # today = get_today(df_5min, date)
    # today = today.iloc[:-2,:]
    # y1, y2, peaks1, peaks2 = get_peaks(today)
    # get_plot(y1, y2, peaks1, peaks2, today)
    # plt.show()
    # port.generate_csv_report()
    # fig = plt.figure(num=None, figsize=(16, 12), dpi=160, facecolor='w', edgecolor='k')
    # plt.plot(port.percent_df['cumprod'], 'bo-')
    # plt.xticks(rotation=45)
    # plt.xlabel('Date-time', fontsize=18)
    # plt.ylabel('Cumulative % change', fontsize=16)
    # plt.savefig(f"./plot/{symbol[:-3]}.jpeg")
    # plt.close(fig)

# store_result = Store_Data()

df = port.percent_df
cum = ((df['Percent_change']/100+1).cumprod()-1)*100
day = []
x = []
for date in sorted(list(set(df.index.date))):
    var = (((df[df.index.date == date]['Percent_change'] / 100 + 1).cumprod() - 1) * 100).values[-1]
    day.append(var)
    x.append(date)
day_wise = pd.DataFrame({'Time':x, 'percent' : day})
day_wise.set_index('Time',drop = True, inplace=True)
cum2 = ((day_wise['percent']/100+1).cumprod()-1)*100

fig, axs = plt.subplots(2)
axs[0].plot(cum)
axs[1].plot(cum2)
axs[0].set_title('Cumulative Profit')
axs[1].set_title('Cumulative Profit Day Wise')
fig.suptitle(f'5min MTF high low : {symbol}')
plt.show()




for date in sorted(list(set(port.order_df.index.date))):
    print(len(port.order_df[date == port.order_df.index.date]))