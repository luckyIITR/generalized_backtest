import pandas as pd
import matplotlib.pyplot as plt


class BuyPortfolio:

    def __init__(self, symbol):
        self.symbol = symbol[:-3]
        self.post_dict = {}
        self.percent_change = []
        self.order_book = []
        self.df_per_change = []
        self.post_dict['Time'] = None
        self.post_dict['Signal'] = ""
        self.post_dict['Price'] = None
        self.post_dict['Pos'] = 0
        self.post_dict['%change'] = 0
        self.order_df = pd.DataFrame()
        self.percent_df = pd.DataFrame()
        self.day_wise = pd.DataFrame()

    def buy(self, bp, time):
        if self.post_dict['Signal'] != 'BUY':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "BUY"
            self.post_dict['Price'] = bp
            self.post_dict['Pos'] = 1
            self.post_dict['%change'] = 0
            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have BUY positions")

    def square_off(self, sp, time):
        if self.post_dict['Signal'] != 'SELL':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "SELL"
            self.post_dict['Price'] = sp
            self.post_dict['Pos'] = 0
            pc = (sp / self.order_book[-1]['Price'] - 1)
            self.percent_change.append(pc * 100)
            self.post_dict['%change'] = pc * 100
            self.df_per_change.append({"Time": time, "%change": pc * 100})
            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have Squared Off positions")

    def check_pos(self):
        return self.post_dict['Pos']

    def generate_dataframes(self):
        if self.post_dict['Pos'] == 0:
            # convert to dataframes
            y = self.order_book.copy()
            self.order_df = pd.DataFrame(y)
            self.order_df.set_index('Time', inplace=True, drop=True)

            z = self.df_per_change
            self.percent_df = pd.DataFrame(z)
            self.percent_df.set_index('Time', inplace=True, drop=True)
            self.percent_df['cumprod'] = ((self.percent_df['%change'] / 100 + 1).cumprod() - 1) * 100

            p_l = []
            m = self.percent_df.index.date
            dates = list(set(m))
            dates = sorted(dates)
            for date in dates:
                t = self.percent_df[self.percent_df.index.date == date].sum()[0]
                p_l.append(t)
            self.day_wise = pd.DataFrame.from_dict({"Time": dates, "%change": p_l}, orient='columns',
                                                   dtype=None, columns=None)
            self.day_wise.set_index("Time", drop=True, inplace=True)
            self.day_wise['cumprod'] = ((self.day_wise['%change']/100 + 1).cumprod() - 1)*100
        else:
            print("First close open positions")

    def generate_results(self):
        self.generate_dataframes()
        gains = 0
        ng = 0
        losses = 0
        nl = 0
        totalR = 1

        for i in self.percent_change:
            if i > 0:
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            totalR = totalR * ((i / 100) + 1)

        totalR = round((totalR - 1) * 100, 2)

        if ng > 0:
            avgGain = gains / ng
            maxR = str(max(self.percent_change))
        else:
            avgGain = 0
            maxR = "undefined"

        if nl > 0:
            avgLoss = losses / nl
            maxL = str(min(self.percent_change))
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
        print("Results for " + self.symbol)
        print("Batting Avg: " + str(battingAvg))
        print("Gain/loss ratio: " + ratio)
        print("Average Gain: " + str(avgGain))
        print("Average Loss: " + str(avgLoss))
        print("Max Return: " + maxR)
        print("Max Loss: " + maxL)
        print("Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%")
        print("###############################################################")
        print()

    def plot_result(self):
        plt.plot(self.percent_df['cumprod'])
        # plt.show()

    def plot_day_wise(self):
        plt.plot(self.day_wise)
        plt.show()

    def get_percent_gain(self):
        return self.percent_df.iloc[-1, -1]

    def generate_csv_report(self):
        writer = pd.ExcelWriter(self.symbol+".xlsx", engine='xlsxwriter')
        self.order_df.to_excel(writer, sheet_name='Order_book')
        self.percent_df.to_excel(writer, sheet_name='%change')
        self.day_wise.to_excel(writer, sheet_name='Day_wise')
        writer.save()


class SellPortfolio:

    def __init__(self, symbol):
        self.symbol = symbol[:-3]
        self.post_dict = {}
        self.percent_change = []
        self.order_book = []
        self.df_per_change = []
        self.post_dict['Time'] = None
        self.post_dict['Signal'] = ""
        self.post_dict['Price'] = None
        self.post_dict['Pos'] = 0
        self.post_dict['%change'] = 0
        self.order_df = pd.DataFrame()
        self.percent_df = pd.DataFrame()
        self.day_wise = pd.DataFrame()

    def sell(self, sp, time):
        if self.post_dict['Signal'] != 'SELL':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "SELL"
            self.post_dict['Price'] = sp
            self.post_dict['Pos'] = -1
            self.post_dict['%change'] = 0
            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have SELL positions")

    def square_off(self, bp, time):
        if self.post_dict['Signal'] != 'BUY':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "BUY"
            self.post_dict['Price'] = bp
            self.post_dict['Pos'] = 0
            pc = (bp / self.order_book[-1]['Price'] - 1)
            self.percent_change.append(pc * 100)
            self.post_dict['%change'] = pc * 100
            self.df_per_change.append({"Time": time, "%change": pc * 100})
            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have BUY positions")

    def check_pos(self):
        return self.post_dict['Pos']

    def generate_dataframes(self):
        if self.post_dict['Pos'] == 0:
            # convert to dataframes
            y = self.order_book.copy()
            self.order_df = pd.DataFrame(y)
            self.order_df.set_index('Time', inplace=True, drop=True)

            z = self.df_per_change
            self.percent_df = pd.DataFrame(z)
            self.percent_df.set_index('Time', inplace=True, drop=True)
            self.percent_df['cumprod'] = ((self.percent_df['%change'] / 100 + 1).cumprod() - 1) * 100

            p_l = []
            m = self.percent_df.index.date
            dates = list(set(m))
            dates = sorted(dates)
            for date in dates:
                t = self.percent_df[self.percent_df.index.date == date].sum()[0]
                p_l.append(t)
            self.day_wise = pd.DataFrame.from_dict({"Time": dates, "%change": p_l}, orient='columns',
                                                   dtype=None, columns=None)
            self.day_wise.set_index("Time", drop=True, inplace=True)
            self.day_wise['cumprod'] = ((self.day_wise['%change']/100 + 1).cumprod() - 1)*100
        else:
            print("Positions  Still open")

    def generate_results(self):
        gains = 0
        ng = 0
        losses = 0
        nl = 0
        totalR = 1

        for i in self.percent_change:
            if i > 0:
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            totalR = totalR * ((i / 100) + 1)

        totalR = round((totalR - 1) * 100, 2)

        if ng > 0:
            avgGain = gains / ng
            maxR = str(max(self.percent_change))
        else:
            avgGain = 0
            maxR = "undefined"

        if nl > 0:
            avgLoss = losses / nl
            maxL = str(min(self.percent_change))
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
        print("Results for " + self.symbol)
        print("Batting Avg: " + str(battingAvg))
        print("Gain/loss ratio: " + ratio)
        print("Average Gain: " + str(avgGain))
        print("Average Loss: " + str(avgLoss))
        print("Max Return: " + maxR)
        print("Max Loss: " + maxL)
        print("Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%")
        print("###############################################################")
        print()

    def plot_result(self):
        plt.plot(self.percent_df['cumprod'])
        # plt.show()

    def plot_day_wise(self):
        plt.plot(self.day_wise)
        plt.show()

    def get_percent_gain(self):
        return self.percent_df.iloc[-1, -1]

    def generate_csv_report(self):
        writer = pd.ExcelWriter(self.symbol+".xlsx", engine='xlsxwriter')
        self.order_df.to_excel(writer, sheet_name='Order_book')
        self.percent_df.to_excel(writer, sheet_name='%change')
        self.day_wise.to_excel(writer, sheet_name='Day_wise')
        writer.save()


class Combine:

    def __init__(self, symbol):
        self.symbol = symbol[:-3]
        self.post_dict = {}
        self.percent_change = []
        self.order_book = []
        self.df_per_change = []
        self.post_dict['Time'] = None
        self.post_dict['Signal'] = ""
        self.post_dict['Price'] = None
        self.post_dict['Pos'] = 0
        self.post_dict['%change'] = 0
        self.order_df = pd.DataFrame()
        self.percent_df = pd.DataFrame()
        self.day_wise = pd.DataFrame()

    def buy(self, bp, time):
        if self.post_dict['Signal'] != 'BUY':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "BUY"
            self.post_dict['Price'] = bp
            self.post_dict['Pos'] = 1
            self.post_dict['%change'] = 0

            if len(self.order_book) :
                pc = (self.order_book[-1]['Price'] /bp - 1)
                self.percent_change.append(pc * 100)
                self.post_dict['%change'] = pc * 100
                self.df_per_change.append({"Time": time, "%change": pc * 100})

            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have BUY positions")

    def sell(self, sp, time):
        if self.post_dict['Signal'] != 'SELL':
            self.post_dict['Time'] = time
            self.post_dict['Signal'] = "SELL"
            self.post_dict['Price'] = sp
            self.post_dict['Pos'] = -1
            pc = (sp / self.order_book[-1]['Price'] - 1)
            self.percent_change.append(pc * 100)
            self.post_dict['%change'] = pc * 100
            self.df_per_change.append({"Time": time, "%change": pc * 100})
            x = self.post_dict.copy()
            self.order_book.append(x)
        else:
            print("You already have SELL positions")

    def check_pos(self):
        return self.post_dict['Pos']

    def generate_dataframes(self):
        # convert to dataframes
        y = self.order_book.copy()
        self.order_df = pd.DataFrame(y)
        self.order_df.set_index('Time', inplace=True, drop=True)
        self.order_df['cumprod'] = ((self.order_df['%change']/100 + 1).cumprod() - 1) * 100
        z = self.df_per_change
        self.percent_df = pd.DataFrame(z)
        self.percent_df.set_index('Time', inplace=True, drop=True)
        self.percent_df['cumprod'] = ((self.percent_df['%change'] / 100 + 1).cumprod() - 1) * 100

        p_l = []
        m = self.percent_df.index.date
        dates = list(set(m))
        dates = sorted(dates)
        for date in dates:
            t = self.percent_df[self.percent_df.index.date == date].sum()[0]
            p_l.append(t)
        self.day_wise = pd.DataFrame.from_dict({"Time": dates, "%change": p_l}, orient='columns',
                                               dtype=None, columns=None)
        self.day_wise.set_index("Time", drop=True, inplace=True)
        self.day_wise['cumprod'] = ((self.day_wise['%change']/100 + 1).cumprod() - 1)*100

    def generate_results(self):
        gains = 0
        ng = 0
        losses = 0
        nl = 0
        totalR = 1

        for i in self.percent_change:
            if i > 0:
                gains += i
                ng += 1
            else:
                losses += i
                nl += 1
            totalR = totalR * ((i / 100) + 1)

        totalR = round((totalR - 1) * 100, 2)

        if ng > 0:
            avgGain = gains / ng
            maxR = str(max(self.percent_change))
        else:
            avgGain = 0
            maxR = "undefined"

        if nl > 0:
            avgLoss = losses / nl
            maxL = str(min(self.percent_change))
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
        print("Results for " + self.symbol)
        print("Batting Avg: " + str(battingAvg))
        print("Gain/loss ratio: " + ratio)
        print("Average Gain: " + str(avgGain))
        print("Average Loss: " + str(avgLoss))
        print("Max Return: " + maxR)
        print("Max Loss: " + maxL)
        print("Total return over " + str(ng + nl) + " trades: " + str(totalR) + "%")
        print("###############################################################")
        print()

    def plot_result(self):
        plt.plot(self.percent_df['cumprod'])
        # plt.show()

    def plot_day_wise(self):
        plt.plot(self.day_wise['cumprod'])

    def get_percent_gain(self):
        return self.percent_df.iloc[-1, -1]

    def generate_csv_report(self):
        writer = pd.ExcelWriter(self.symbol+".xlsx", engine='xlsxwriter')
        self.order_df.to_excel(writer, sheet_name='Order_book')
        self.percent_df.to_excel(writer, sheet_name='%change')
        self.day_wise.to_excel(writer, sheet_name='Day_wise')
        writer.save()