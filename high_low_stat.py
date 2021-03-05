import numpy as np
import yfinance as yf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from Portfolio2 import BuyPortfolio, Store_Data, SellPortfolio
# import os
# import sqlite3
import pandas as pd
import datetime as dt
from finta import TA
from plotly.offline import plot
# plt.ioff()


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
    daily = yf.download(tickers=symbol, interval="5m",start=dt.datetime(2021,2,3,9,15),end = dt.datetime(2021,2,23))
    # daily = yf.download(tickers=symbol, interval="5m", period=f"{str(tp)}d" )
    daily.index = daily.index.tz_localize(None)
    daily.drop(["Adj Close", 'Volume'], axis=1, inplace=True)
    return daily


def get_dates(symbol, tp):
    daily = yf.download(tickers=symbol, interval="60m",start=dt.datetime(2021,1,15,9,15),end = dt.datetime(2021,3,23))
    # daily = yf.download(tickers=symbol, interval="60m", period=f"{str(tp)}d")
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


def main(symbol):
    # symbol = "HCLTECH.NS"
    backtest_tp = 10
    df_5min = get_intra_data(symbol, backtest_tp)
    df_hour = get_dates(symbol, backtest_tp+5)
    df_hour['ema21'] = TA.EMA(df_hour, period=21)
    df_hour['ema8'] = TA.EMA(df_hour, period=8)
    df_hour = df_hour.iloc[21:, :].copy()
    df_hour['signal'] = [
        1 if df_hour.loc[e, 'ema8'] - df_hour.loc[e, 'ema21'] > 0 and df_hour.loc[e, 'Close'] > df_hour.loc[
            e, 'ema8'] else 0 for e in df_hour.index]
    df_hour['signal'] = df_hour['signal'].shift(1)
    df_5min = pd.concat([df_5min, df_hour['signal']], axis=1)
    df_5min = df_5min.ffill()
    df_5min.dropna(inplace=True)
    # df_5min = df_5min[df_5min.index.time >= dt.datetime(2020, 2, 2, 10, 15).time()]
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

    if len(port.order_book) != 0 :
        port.generate_dataframes()
        store_result.append_data(port.generate_results())
        store_result.day_wise_result(port.get_day_wise().rename(columns={'%change': f"{symbol[:-3]}"}))


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



store_result = Store_Data()

# tickers = ['ADANIPORTS.NS',
#            'ASIANPAINT.NS',
#            'AXISBANK.NS',
#            'BAJAJ-AUTO.NS',
#            'BAJAJFINSV.NS',
#            'BAJFINANCE.NS',
#            'BHARTIARTL.NS',
#            'BPCL.NS',
#            'BRITANNIA.NS',
#            'CIPLA.NS',
#            'COALINDIA.NS',
#            'DIVISLAB.NS',
#            'DRREDDY.NS',
#            'EICHERMOT.NS',
#            'GAIL.NS',
#            'GRASIM.NS',
#            'HCLTECH.NS',
#            'HDFC.NS',
#            'HDFCBANK.NS',
#            'HDFCLIFE.NS',
#            'HEROMOTOCO.NS',
#            'HINDALCO.NS',
#            'HINDUNILVR.NS',
#            'ICICIBANK.NS',
#            'INDUSINDBK.NS',
#            'INFY.NS',
#            'IOC.NS',
#            'ITC.NS',
#            'JSWSTEEL.NS',
#            'KOTAKBANK.NS',
#            'LT.NS',
#            'M&M.NS',
#            'MARUTI.NS',
#            'NESTLEIND.NS',
#            'NTPC.NS',
#            'ONGC.NS',
#            'POWERGRID.NS',
#            'RELIANCE.NS',
#            'SBILIFE.NS',
#            'SBIN.NS',
#            'SUNPHARMA.NS',
#            'TATAMOTORS.NS',
#            'TATASTEEL.NS',
#            'TCS.NS',
#            'TECHM.NS',
#            'TITAN.NS',
#            'ULTRACEMCO.NS',
#            'UPL.NS',
#            'WIPRO.NS']

tickers = ['ADANIPORTS.NS',
           'ASIANPAINT.NS',
           'AXISBANK.NS',
           'BAJAJ-AUTO.NS',
           'BHARTIARTL.NS',
           'BPCL.NS',
           'CIPLA.NS',
           'COALINDIA.NS',
           'DIVISLAB.NS',
           'DRREDDY.NS',
           'GAIL.NS',
           'GRASIM.NS',
           'HCLTECH.NS',
           'HDFC.NS',
           'HDFCBANK.NS',
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
           'NTPC.NS',
           'ONGC.NS',
           'POWERGRID.NS',
           'RELIANCE.NS',
           'SBIN.NS',
           'SUNPHARMA.NS',
           'TATAMOTORS.NS',
           'TATASTEEL.NS',
           'TCS.NS',
           'TECHM.NS',
           'TITAN.NS',
           'UPL.NS',
           'WIPRO.NS']
symbol = tickers[1]

tickers = ['AVANTIFEED.NS',
 'TIINDIA.NS',
 'EXIDEIND.NS',
 'LINDEINDIA.NS',
 'CEATLTD.NS',
 'PVR.NS',
 'SUVENPHAR.NS',
 'SIS.NS',
 'VTL.NS',
 'DHANUKA.NS',
 'STAR.NS',
 'VAKRANGEE.NS',
 'IBREALEST.NS',
 'BLUEDART.NS',
 'CREDITACC.NS',
 'KNRCON.NS',
 'AEGISCHEM.NS',
 'NATIONALUM.NS',
 'CARERATING.NS',
 'ALOKINDS.NS',
 'VENKEYS.NS',
 'IDFCFIRSTB.NS',
 'GMRINFRA.NS',
 'SOUTHBANK.NS',
 'CSBBANK.NS',
 'SCHAEFFLER.NS',
 'GESHIP.NS',
 'TVSMOTOR.NS',
 'MINDACORP.NS',
 'HAL.NS',
 'SOBHA.NS',
 'SCI.NS',
 'RCF.NS',
 'FINCABLES.NS',
 'GMDCLTD.NS',
 'GRINDWELL.NS',
 'IDFC.NS',
 'FINPIPE.NS',
 'MAHINDCIE.NS',
 'MINDAIND.NS',
 'LICHSGFIN.NS',
 'EQUITAS.NS',
 'ZENSARTECH.NS',
 'BDL.NS',
 'VIPIND.NS',
 'ALKYLAMINE.NS',
 'GUJGASLTD.NS',
 'JINDALSTEL.NS',
 'DELTACORP.NS',
 'CAPLIPOINT.NS',
 'HONAUT.NS',
 'HFCL.NS',
 'KRBL.NS',
 'IEX.NS',
 'INDIACEM.NS',
 'JCHAC.NS',
 'SPICEJET.NS',
 'WABCOINDIA.NS',
 'PTC.NS',
 'PERSISTENT.NS',
 'RAMCOCEM.NS',
 'SUPRAJIT.NS',
 'BRIGADE.NS',
 'BALRAMCHIN.NS',
 'KOLTEPATIL.NS',
 'MPHASIS.NS',
 'BIRLACORPN.NS',
 'GAEL.NS',
 'SEQUENT.NS',
 'POLYPLEX.NS',
 'CUMMINSIND.NS',
 'GMMPFAUDLR.NS',
 'NILKAMAL.NS',
 'BATAINDIA.NS',
 'RAYMOND.NS',
 'ADANIENT.NS',
 'BALMLAWRIE.NS',
 'NH.NS',
 'ESABINDIA.NS',
 'CANFINHOME.NS',
 'GODREJPROP.NS',
 'AUBANK.NS',
 'SUZLON.NS',
 'SUDARSCHEM.NS',
 'EIHOTEL.NS',
 'KALPATPOWR.NS',
 'SAIL.NS',
 'IOLCP.NS',
 'SUNTECK.NS',
 'UJJIVANSFB.NS',
 'ASTRAL.NS',
 'RATNAMANI.NS',
 'MASFIN.NS',
 'CUB.NS',
 'SOLARA.NS',
 'APLAPOLLO.NS',
 'JUSTDIAL.NS',
 'TASTYBITE.NS',
 'BSE.NS',
 'WELSPUNIND.NS',
 'BLISSGVS.NS',
 'GNFC.NS',
 'TIMKEN.NS',
 'RAIN.NS',
 'MGL.NS',
 'SYNGENE.NS',
 'SUPPETRO.NS',
 'CHOLAFIN.NS',
 'JAMNAAUTO.NS',
 'COROMANDEL.NS',
 'STLTECH.NS',
 'KSB.NS',
 'UFLEX.NS',
 'JBCHEPHARM.NS',
 'DCBBANK.NS',
 'VAIBHAVGBL.NS',
 'HUDCO.NS',
 'PRAJIND.NS',
 'CHALET.NS',
 'SYMPHONY.NS',
 'ZYDUSWELL.NS',
 'BLUESTARCO.NS',
 'TRIDENT.NS',
 'BHARATFORG.NS',
 'BAJAJCON.NS',
 'VSTIND.NS',
 'CENTURYTEX.NS',
 'CRISIL.NS',
 'TATASTLBSL.NS',
 'CENTRALBK.NS',
 'JSLHISAR.NS',
 'CDSL.NS',
 'APLLTD.NS',
 'MOIL.NS',
 'IIFL.NS',
 'BOMDYEING.NS',
 'ELGIEQUIP.NS',
 'BEML.NS',
 'MFSL.NS',
 'KTKBANK.NS',
 'SCHNEIDER.NS',
 'SRF.NS',
 'ORIENTELEC.NS',
 'SHILPAMED.NS',
 'SHOPERSTOP.NS',
 'STARCEMENT.NS',
 'GODFRYPHLP.NS',
 'TATAINVEST.NS',
 'RADICO.NS',
 'PGHL.NS',
 'JKLAKSHMI.NS',
 'CERA.NS',
 'KAJARIACER.NS',
 'JMFINANCIL.NS',
 'MMTC.NS',
 'MHRIL.NS',
 'JYOTHYLAB.NS',
 'IOB.NS',
 'ADVENZYMES.NS',
 'BAYERCROP.NS',
 'BALKRISIND.NS',
 'CASTROLIND.NS',
 'RALLIS.NS',
 'OMAXE.NS',
 'TORNTPOWER.NS',
 'PRSMJOHNSN.NS',
 'TRENT.NS',
 'LAOPALA.NS',
 'MAHSCOOTER.NS',
 'EPL.NS',
 'GLAXO.NS',
 'MCX.NS',
 'AFFLE.NS',
 'IDBI.NS',
 'INGERRAND.NS',
 '3MINDIA.NS',
 'TATACOFFEE.NS',
 'KEI.NS',
 'CARBORUNIV.NS',
 'AJANTPHARM.NS',
 'DBCORP.NS',
 'KSCL.NS',
 'INDHOTEL.NS',
 'IPCALAB.NS',
 'VINATIORGA.NS',
 'RECLTD.NS',
 'UJJIVAN.NS',
 'NCC.NS',
 'FSL.NS',
 'NFL.NS',
 'FLUOROCHEM.NS',
 'DISHTV.NS',
 'AMARAJABAT.NS',
 'INOXLEISUR.NS',
 'TCNSBRANDS.NS',
 'JAGRAN.NS',
 'SWSOLAR.NS',
 'NESCO.NS',
 'MINDTREE.NS',
 'MAHSEAMLES.NS',
 'LUXIND.NS',
 'NOCIL.NS',
 'HERITGFOOD.NS',
 'ENGINERSIN.NS',
 'SANOFI.NS',
 'INDIAMART.NS',
 'HINDCOPPER.NS',
 'HUHTAMAKI.NS',
 'IRB.NS',
 'GULFOILLUB.NS',
 'NAVNETEDUL.NS',
 'SJVN.NS',
 'MAHLOG.NS',
 'L&TFH.NS',
 'NETWORK18.NS',
 'CCL.NS',
 'ICRA.NS',
 'JSWENERGY.NS',
 'ORIENTCEM.NS',
 'PSPPROJECT.NS',
 'LEMONTREE.NS',
 'FINEORG.NS',
 'UCOBANK.NS',
 'ASTERDM.NS',
 'RITES.NS',
 'JKCEMENT.NS',
 'HEG.NS',
 'SHRIRAMCIT.NS',
 'KANSAINER.NS',
 'METROPOLIS.NS',
 'FDC.NS',
 'JSL.NS',
 'MAHABANK.NS',
 'JKTYRE.NS',
 'KARURVYSYA.NS',
 'ERIS.NS',
 'ALEMBICLTD.NS',
 'PNBHOUSING.NS',
 'IFBIND.NS',
 'HATHWAY.NS',
 'POLYCAB.NS',
 'CHAMBLFERT.NS',
 'TATAPOWER.NS',
 'JTEKTINDIA.NS',
 'BAJAJELEC.NS',
 'WESTLIFE.NS',
 'INDIANB.NS',
 'IBULHSGFIN.NS',
 'JAICORPLTD.NS',
 'GARFIBRES.NS',
 'OIL.NS',
 'LALPATHLAB.NS',
 'SKFINDIA.NS',
 'J&KBANK.NS',
 'ESCORTS.NS',
 'SPARC.NS',
 'NBCC.NS',
 'AMBER.NS',
 'TV18BRDCST.NS',
 'TATAELXSI.NS',
 'VMART.NS',
 'SOLARINDS.NS',
 'HSCL.NS',
 'COFORGE.NS',
 'BEL.NS',
 'DCAL.NS',
 'REDINGTON.NS',
 'GPPL.NS',
 'WOCKPHARMA.NS',
 'JKPAPER.NS',
 'M&MFIN.NS',
 'GRANULES.NS',
 'GALAXYSURF.NS',
 'DEEPAKNTR.NS',
 'APOLLOTYRE.NS',
 'VARROC.NS',
 'DALBHARAT.NS',
 'SWANENERGY.NS',
 'EIDPARRY.NS',
 'ZEEL.NS',
 'ORIENTREF.NS',
 'GEPIL.NS',
 'LAXMIMACH.NS',
 'GRAPHITE.NS',
 'MRF.NS',
 'GSFC.NS',
 'AARTIDRUGS.NS',
 'BHARATRAS.NS',
 'DIXON.NS',
 'VOLTAS.NS',
 'ASHOKLEY.NS',
 'PHILIPCARB.NS',
 'TATACHEM.NS',
 'INDOCO.NS',
 'SWARAJENG.NS',
 'DCMSHRIRAM.NS',
 'IRCON.NS',
 'FCONSUMER.NS',
 'PNCINFRA.NS',
 'THYROCARE.NS',
 'HATSUN.NS',
 'CENTURYPLY.NS',
 'RVNL.NS',
 'QUESS.NS',
 'SRTRANSFIN.NS',
 'ABCAPITAL.NS',
 'BSOFT.NS',
 'POWERINDIA.NS',
 'TEAMLEASE.NS',
 'GUJALKALI.NS',
 'EDELWEISS.NS',
 'GREAVESCOT.NS',
 'ECLERX.NS',
 'NLCINDIA.NS',
 'VRLLOG.NS',
 'GRSE.NS',
 'COCHINSHIP.NS',
 'GHCL.NS',
 'IIFLWAM.NS',
 'HEIDELBERG.NS',
 'JINDALSAW.NS',
 'YESBANK.NS',
 'KEC.NS',
 'JUBLFOOD.NS',
 'TVTODAY.NS',
 'FORTIS.NS',
 'SONATSOFTW.NS',
 'POLYMED.NS',
 'TCIEXP.NS',
 'ASHOKA.NS',
 'LAURUSLABS.NS',
 'BASF.NS',
 'CGCL.NS',
 'GSPL.NS',
 'SUNDARMFIN.NS',
 'CYIENT.NS',
 'CHENNPETRO.NS',
 'MIDHANI.NS',
 'PRESTIGE.NS',
 'IRCTC.NS',
 'WELCORP.NS',
 'DBL.NS',
 'AAVAS.NS']
tickers  = ['ADANIPORTS.NS',
 'APOLLOTYRE.NS',
 'ASHOKLEY.NS',
 'AXISBANK.NS',
 'BAJFINANCE.NS',
 'BAJAJFINSV.NS',
 'BANDHANBNK.NS',
 'BANKBARODA.NS',
 'BHEL.NS',
 'BPCL.NS',
 'CANFINHOME.NS',
 'CANBK.NS',
 'CHOLAFIN.NS',
 'COFORGE.NS',
 'DLF.NS',
 'ESCORTS.NS',
 'FEDERALBNK.NS',
 'GODREJPROP.NS',
 'GRASIM.NS',
 'HINDALCO.NS',
 'HDFC.NS',
 'ICICIBANK.NS',
 'ICICIPRULI.NS',
 'IDFCFIRSTB.NS',
 'IBULHSGFIN.NS',
 'INDUSINDBK.NS',
 'JSWSTEEL.NS',
 'JINDALSTEL.NS',
 'L&TFH.NS',
 'LICHSGFIN.NS',
 'M&MFIN.NS',
 'MANAPPURAM.NS',
 'MARUTI.NS',
 'MFSL.NS',
 'MOTHERSUMI.NS',
 'MUTHOOTFIN.NS',
 'NMDC.NS',
 'NAM-INDIA.NS',
 'PEL.NS',
 'PFC.NS',
 'RBLBANK.NS',
 'RADICO.NS',
 'SRTRANSFIN.NS',
 'SBIN.NS',
 'SAIL.NS',
 'TATAMOTORS.NS',
 'TATASTEEL.NS',
 'UJJIVAN.NS',
 'VEDL.NS',
 'IDEA.NS']


for symbol in tickers:
    main(symbol)

result, day_wise = store_result.gen_pd(len(tickers))
store_result.get_csv()

# store_result.day_wise

# t = ((day_wise / 100 + 1).cumprod() - 1) * 100

import plotly.express as px
import plotly.graph_objects as go



def sell(symbol):
    # symbol = "CIPLA.NS"
    backtest_tp = 10 # no use
    df_5min = get_intra_data(symbol, backtest_tp)
    df_hour = get_dates(symbol, backtest_tp+5)
    df_hour['ema21'] = TA.EMA(df_hour, period=21)
    df_hour['ema8'] = TA.EMA(df_hour, period=8)
    df_hour = df_hour.iloc[21:, :].copy()
    df_hour['signal'] = [
        1 if df_hour.loc[e, 'ema8'] - df_hour.loc[e, 'ema21'] < 0 and df_hour.loc[e, 'Close'] < df_hour.loc[
            e, 'ema8'] else 0 for e in df_hour.index]
    df_hour['signal'] = df_hour['signal'].shift(1)
    df_5min = pd.concat([df_5min, df_hour['signal']], axis=1)
    df_5min = df_5min.ffill()
    df_5min.dropna(inplace=True)
    # df_5min = df_5min[df_5min.index.time >= dt.datetime(2020,2,2,10,15).time()]
    port = SellPortfolio(symbol)

    dates = sorted(list(set(df_hour.index.date)))
    for date in dates:
        today = get_today(df_5min, date)
        for e in today.index:
            if e == today.index[0]:
                continue
            y1, y2, peaks1, peaks2 = get_peaks(today.loc[:e, ])
            if len(peaks1) == 0 or len(peaks2) == 0:
                continue

            if today.loc[e, 'signal'] == 1 and today.loc[e, 'Low'] < y2[peaks2[-1]]*-1 and port.check_pos() == 0:
                port.sell(y2[peaks2[-1]]*-1, e)

            elif today.loc[e, 'High'] > y1[peaks1[-1]] and port.check_pos() == -1:
                port.square_off(y1[peaks1[-1]], e)


            if port.check_pos() == -1 and e.time() == dt.datetime(2020, 2, 2, 15, 25).time():
                port.square_off(today.loc[e, 'Open'], e)
    if len(port.order_book) != 0:
        port.generate_dataframes()
        store_result.append_data(port.generate_results())
        store_result.day_wise_result(port.get_day_wise().rename(columns={'%change': f"{symbol[:-3]}"}))

    #
    # today = get_today(df_5min, date)
    # df = today
    # y1, y2, peaks1, peaks2 = get_peaks(today)
    # get_plot(y1, y2, peaks1, peaks2, today)

store_result = Store_Data()

# tickers = ['ADANIPORTS.NS',
#            'ASIANPAINT.NS',
#            'AXISBANK.NS',
#            'BAJAJ-AUTO.NS',
#            'BAJAJFINSV.NS',
#            'BAJFINANCE.NS',
#            'BHARTIARTL.NS',
#            'BPCL.NS',
#            'BRITANNIA.NS',
#            'CIPLA.NS',
#            'COALINDIA.NS',
#            'DIVISLAB.NS',
#            'DRREDDY.NS',
#            'EICHERMOT.NS',
#            'GAIL.NS',
#            'GRASIM.NS',
#            'HCLTECH.NS',
#            'HDFC.NS',
#            'HDFCBANK.NS',
#            'HDFCLIFE.NS',
#            'HEROMOTOCO.NS',
#            'HINDALCO.NS',
#            'HINDUNILVR.NS',
#            'ICICIBANK.NS',
#            'INDUSINDBK.NS',
#            'INFY.NS',
#            'IOC.NS',
#            'ITC.NS',
#            'JSWSTEEL.NS',
#            'KOTAKBANK.NS',
#            'LT.NS',
#            'M&M.NS',
#            'MARUTI.NS',
#            'NESTLEIND.NS',
#            'NTPC.NS',
#            'ONGC.NS',
#            'POWERGRID.NS',
#            'RELIANCE.NS',
#            'SBILIFE.NS',
#            'SBIN.NS',
#            'SUNPHARMA.NS',
#            'TATAMOTORS.NS',
#            'TATASTEEL.NS',
#            'TCS.NS',
#            'TECHM.NS',
#            'TITAN.NS',
#            'ULTRACEMCO.NS',
#            'UPL.NS',
#            'WIPRO.NS']

tickers = ['ADANIPORTS.NS',
           'ASIANPAINT.NS',
           'AXISBANK.NS',
           'BAJAJ-AUTO.NS',
           'BHARTIARTL.NS',
           'BPCL.NS',
           'CIPLA.NS',
           'COALINDIA.NS',
           'DIVISLAB.NS',
           'DRREDDY.NS',
           'GAIL.NS',
           'GRASIM.NS',
           'HCLTECH.NS',
           'HDFC.NS',
           'HDFCBANK.NS',
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
           'NTPC.NS',
           'ONGC.NS',
           'POWERGRID.NS',
           'RELIANCE.NS',
           'SBIN.NS',
           'SUNPHARMA.NS',
           'TATAMOTORS.NS',
           'TATASTEEL.NS',
           'TCS.NS',
           'TECHM.NS',
           'TITAN.NS',
           'UPL.NS',
           'WIPRO.NS']
symbol = tickers[1]



for symbol in tickers:
    sell(symbol)

result, day_wise = store_result.gen_pd(len(tickers))
store_result.get_csv()