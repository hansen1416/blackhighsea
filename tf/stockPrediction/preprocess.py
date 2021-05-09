import os
import sys

import numpy as np
import pandas as pd
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler

DATASET_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
MODEL_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def get_stock_data(ticker, shape=13) -> pd.DataFrame:

    df = pd.read_csv(os.path.join(DATASET_DIR, ticker + '.csv'), \
        index_col='date', parse_dates=False)

    return df[['close', 'high', 'low', 'open', 'volume', 'increase_in_vol',
       'increase_in_close', 'up_ratio', 'up_ratio_next', 'moving_av50',
       'moving_av30', 'moving_av20', 'moving_av10', 'moving_av5',
       'ma50_close_ratio', 'ma30_close_ratio', 'ma20_close_ratio',
       'ma10_close_ratio', 'ma5_close_ratio', 'macd']]

def scale_data(df):

    df = df.drop('p_close', axis=1)

    # print(df)

    x = df
    y = df[['close']].shift(-1)
    
    # drop the last row, becasue we shifted the price by 1
    y.drop(y.tail(1).index, inplace=True)
    x.drop(x.tail(1).index, inplace=True)

    x_scaler = MinMaxScaler(feature_range=(0,1)).fit(x)
    y_scaler = MinMaxScaler(feature_range=(0,1)).fit(y)

    features = x_scaler.transform(x)
    target = y_scaler.transform(y)

    # print(features.shape)

    return features, target, x_scaler, y_scaler

def lstm_preprocess_data1(df):

    df = df[df['close'] != 0]
    df = df.sort_values(by='date', ascending=True)
    
    # we will just train on all of the data
    x_train_raw = df.values
    y_train_raw = df[['close']].values

    sc_x = MinMaxScaler(feature_range=(0,1))
    sc_y = MinMaxScaler(feature_range=(0,1))
    # todo, define a global y scaler
    x_train_scaled = sc_x.fit_transform(x_train_raw)
    y_train_scaled = sc_y.fit_transform(y_train_raw)

    x_train = []
    y_train = []

    for i in range(50, len(x_train_scaled)):
        x_train.append(x_train_scaled[i-50:i,:])
        y_train.append(y_train_scaled[i,:])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train

def lstm_preprocess_data(df):

    df = df[df['close'] != 0]
    df = df.sort_values(by='date', ascending=True)
    
    # we will just train on all of the data
    x_train_raw = df.values
    y_train_raw = df[['close']].values

    x_scaler = MinMaxScaler(feature_range=(0,1)).fit(x_train_raw)
    y_scaler = MinMaxScaler(feature_range=(0,1)).fit(y_train_raw)
    # todo, define a global y scaler
    x_train_scaled = x_scaler.transform(x_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw)

    x_train = []
    y_train = []

    for i in range(50, len(x_train_scaled)):
        x_train.append(x_train_scaled[i-50:i,:])
        y_train.append(y_train_scaled[i,:])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    dump(x_scaler, open(os.path.join(MODEL_DIR, 'x_scaler.pkl'), 'wb'))
    dump(y_scaler, open(os.path.join(MODEL_DIR, 'y_scaler.pkl'), 'wb'))

    return x_train, y_train, x_scaler, y_scaler

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Please pass ticker"

    ticker = sys.argv[1]

    df = get_stock_data(ticker)

    print(df.columns)

    x, y, _, _ = lstm_preprocess_data(df)

