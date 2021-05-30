from datetime import datetime
import os
import sys

import pandas as pd
import talib

def save_stock_df(df, history_dir, stock_code) -> None:
    file_path = os.path.join(history_dir, stock_code+'.csv')
    df.to_csv(file_path)
    print('stock {0} data saved to {1}'.format(stock_code, file_path))

def expand_basic_data(df) -> pd.DataFrame:
    if df.empty:
        return df
    # add increase_in_vol, increase_in_close, moving_av
    i=1
    rate_increase_in_vol=[0]
    rate_increase_in_close=[0]
    up_ratio=[0]

    while i<len(df):
        rate_increase_in_vol.append(df.iloc[i]['volume']-df.iloc[i-1]['volume'])
        rate_increase_in_close.append(df.iloc[i]['close']-df.iloc[i-1]['close'])
        up_ratio.append((df.iloc[i]['close']-df.iloc[i-1]['close'])/df.iloc[i-1]['close'])
        i+=1

    df['increase_in_vol']=rate_increase_in_vol
    df['increase_in_close']=rate_increase_in_close
    df['up_ratio'] = up_ratio
    df['up_ratio_next'] = df['up_ratio'].shift(-1)
    df['moving_av50'] = df['close'].rolling(window=50,min_periods=0).mean()
    df['moving_av30'] = df['close'].rolling(window=30,min_periods=0).mean()
    df['moving_av20'] = df['close'].rolling(window=20,min_periods=0).mean()
    df['moving_av10'] = df['close'].rolling(window=10,min_periods=0).mean()
    df['moving_av5'] = df['close'].rolling(window=5,min_periods=0).mean()
    
    df['ma50_close_ratio'] = df['close'] / df['moving_av50']
    df['ma30_close_ratio'] = df['close'] / df['moving_av30']
    df['ma20_close_ratio'] = df['close'] / df['moving_av20']
    df['ma10_close_ratio'] = df['close'] / df['moving_av10']
    df['ma5_close_ratio'] = df['close'] / df['moving_av5']

    macd, macdsignal, macdhist = talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

    df['macd'] = macd

    df = df.fillna(0)
    
    return df

def get_stock_basic_data(stock_code, start_date, end_date, history_dir, to_file=False, expand_data=False) -> pd.DataFrame:
    individual_stock_url = 'http://quotes.money.163.com/service/chddata.html?code={0}&start={1}&end={2}&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;VOTURNOVER;'
    rename_columns = {'日期': 'date', '收盘价': 'close', '最高价': 'high', '最低价':'low', '开盘价':'open', '前收盘':'p_close', '成交量': 'volume'}

    if stock_code.startswith('6'):
        code_url = '0' + stock_code
    else:
        code_url = '1' + stock_code

    df = pd.read_csv(individual_stock_url.format(code_url, start_date, end_date), encoding='GB2312')

    df = df.rename(columns=rename_columns)

    df = df[rename_columns.values()]
    # we order by date asc, 
    # we need `increase_in_vol`, `increase_in_close`, `moving_av` of the last day for predicting
    # since we are predicting the price of tomorrow
    df = df.sort_values(by='date', ascending=True)

    df = df.fillna(0)

    df = df[(df['close'] != 0) & (df['open'] != 0)]

    df = df.set_index('date')

    if expand_data:
        df = expand_basic_data(df)

    if to_file:
        save_stock_df(df, history_dir, stock_code)
    
    return df


if __name__ == "__main__":

    assert len(sys.argv) == 2, "Please pass ticker"

    ticker = sys.argv[1]

    start_date = '20100101'

    date_string_format = "%Y%m%d"

    now_date = datetime.today().strftime(date_string_format)

    history_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

    get_stock_basic_data(ticker, start_date, now_date, history_dir, to_file=True, expand_data=True)