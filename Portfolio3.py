import pandas as pd
import numpy as np


class Combine:

    def __init__(self, symbol):
        self.symbol = symbol
        self.post_dict = {}
        self.order_book = []
        self.post_dict['Time'] = None
        self.post_dict['Signal'] = ""
        self.post_dict['Price'] = None
        self.post_dict['Pos'] = 0
        self.order_df = pd.DataFrame()
        self.result = {}
        self.percent_change = {}
        self.percent_df = pd.DataFrame()

    def buy(self, bp, time):
        # if self.post_dict['Signal'] != 'BUY':
        self.post_dict['Time'] = time
        self.post_dict['Signal'] = "BUY"
        self.post_dict['Price'] = bp
        if self.check_even():
            self.post_dict['Pos'] = 1
        else :
            self.post_dict['Pos'] = 0
        x = self.post_dict.copy()
        self.order_book.append(x)
        # else:
        #     print("You already have BUY positions")

    def sell(self, sp, time):
        # if self.post_dict['Signal'] != 'SELL':
        self.post_dict['Time'] = time
        self.post_dict['Signal'] = "SELL"
        self.post_dict['Price'] = sp
        if self.check_even():
            self.post_dict['Pos'] = -1
        else:
            self.post_dict['Pos'] = 0
        x = self.post_dict.copy()
        self.order_book.append(x)
        # else:
        #     print("You already have SELL positions")

    def check_pos(self):
        return self.post_dict['Pos']

    def generate_dataframes(self):
        # convert to dataframes
        y = self.order_book.copy()
        self.order_df = pd.DataFrame(y)
        self.order_df.set_index('Time', inplace=True, drop=True)
        self.order_df.dropna(inplace=True)
        if len(self.order_df) % 2 == 0:
            i = 0
            for e in self.order_df.index:
                i = i + 1
                if i % 2 == 0:
                    print(self.order_df.loc[e, 'Signal'])
                    if self.order_df.loc[e, 'Signal'] == 'SELL':
                        pc = (self.order_df.loc[e, 'Price'] - self.order_df.loc[self.order_df.index[i - 2], 'Price']) / \
                             self.order_df.loc[self.order_df.index[i - 2], 'Price'] * 100
                        self.percent_change[self.order_df.index[i - 2]] = pc

                    else:
                        pc = (self.order_df.loc[self.order_df.index[i - 2], 'Price'] - self.order_df.loc[e, 'Price']) / \
                             self.order_df.loc[e, 'Price'] * 100
                        self.percent_change[self.order_df.index[i - 2]] = pc
        self.percent_df = pd.DataFrame({'Time' : self.percent_change.keys(), 'Percent_change': self.percent_change.values()})
        self.percent_df.set_index('Time', inplace=True, drop=True)

    def generate_results(self):
        gains = 0
        ng = 0
        losses = 0
        nl = 0
        totalR = 1
        for i in self.percent_df.values.copy():
            if i > 0:
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            totalR = totalR * ((i / 100) + 1)

        totalR = np.round((totalR - 1) * 100, 2)

        if ng > 0:
            avgGain = gains / ng
            maxR = str(max(self.percent_change.values()))
        else:
            avgGain = 0
            maxR = "undefined"

        if nl > 0:
            avgLoss = losses / nl
            maxL = str(min(self.percent_change.values()))
            ratio = str(-avgGain / avgLoss)
        else:
            avgLoss = 0
            maxL = "undefined"
            ratio = "inf"

        if ng > 0 or nl > 0:
            battingAvg = ng / (ng + nl)
        else:
            battingAvg = 0
        print()
        print("###############################################################")
        print(f"From {self.percent_df.index[0]} to {self.percent_df.index[-1]}")
        print("Results for " + self.symbol)
        print("Batting Avg: " + str(battingAvg))
        print("Gain/loss ratio: " + ratio)
        print("Average Gain: " + str(avgGain[0]))
        print("Average Loss: " + str(avgLoss[0]))
        print("Max Return: " + maxR)
        print("Max Loss: " + maxL)
        print("Total return over " + str(ng + nl) + " trades: " + str(totalR[0]) + "%")
        print("###############################################################")
        print()
        self.result["symbol"] = self.symbol
        self.result['Batting Avg'] = battingAvg
        self.result['Gain/loss ratio'] = ratio
        self.result['Average Gain'] = avgGain[0]
        self.result['Average Loss'] = avgLoss[0]
        self.result['NOT'] = ng + nl
        self.result['Max Return'] = maxR
        self.result['Max Loss'] = maxL
        self.result['Total return'] = totalR[0]
        return self.result.copy()

    def check_even(self):
        return len(self.order_book) % 2 == 0