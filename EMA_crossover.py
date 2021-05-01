from portfolio_file_gen import Portfolio
import os
import pandas as pd

cwd = r"F:\Database\Stock_Data\2018-2020"


def get_data(stock):
    stock_file = stock + ".csv"
    stock_file_loc = os.path.join(cwd, stock_file)
    data = pd.read_csv(stock_file_loc)
    data = data.resample('5T').agg({'Open': 'first',
                                    'High': 'max',
                                    'Low': 'min',
                                    'Close': 'last'})
    data = data[data.index.month == data.index.date[-1].month]
    data.dropna(inplace=True)


symbol = "INDUSINDBK"