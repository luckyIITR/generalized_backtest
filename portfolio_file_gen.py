import pandas as pd
import numpy as np


class Portfolio:

    def __init__(self, symbol):
        self.symbol = symbol
        self.status = 0
        self.order_book = pd.DataFrame()

    def buy(self, bp, time):
        self.order_book = pd.concat([self.order_book, pd.DataFrame({'Price': [bp], 'Status': ['BUY'], 'Time': [time]})],
                                    axis=1)

    def sell(self, sp, time):
        self.order_book = pd.concat(
            [self.order_book, pd.DataFrame({'Price': [sp], 'Status': ['SELL'], 'Time': [time]})], axis=1)

    def save_data(self):
        self.order_book.to_csv("C:\Users\lucky\PycharmProjects\generalized_backtest\Saved_data\\"+self.symbol+".csv")