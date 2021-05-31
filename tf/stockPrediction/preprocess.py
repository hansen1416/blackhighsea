import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, datasets_dir) -> pd.DataFrame:

    df = pd.read_csv(os.path.join(datasets_dir, ticker + '.csv'), \
        index_col='date', parse_dates=False)

    return df[['close', 'high', 'low', 'open', 'volume', 'up_ratio', \
                'increase_in_vol', 'moving_av50', 'moving_av30', \
                'moving_av20', 'moving_av10', 'moving_av5', 'macd']]

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

def lstm_preprocess_data(df, ticker, scaler_dir):

    """
    preprocess the tech data
    use 50 days window, and scale data to [0,1]
    """
    df = df[df['close'] != 0]
    df = df.sort_values(by='date', ascending=True)
    
    # close price is the target
    x_train_raw = df.values
    y_train_raw = df[['close']].values
    # min-max scaler
    sc_x = MinMaxScaler(feature_range=(0,1)).fit(x_train_raw)
    sc_y = MinMaxScaler(feature_range=(0,1)).fit(y_train_raw)
    
    x_train_scaled = sc_x.transform(x_train_raw)
    y_train_scaled = sc_y.transform(y_train_raw)

    x_train = []
    y_train = []
    # past 50 days window
    for i in range(50, len(x_train_scaled)):
        x_train.append(x_train_scaled[i-50:i,:])
        y_train.append(y_train_scaled[i,:])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    # save scaler to file
    pickle.dump(sc_x, open(os.path.join(scaler_dir, 'x_scaler_' + ticker + '.pkl'), 'wb'))
    pickle.dump(sc_y, open(os.path.join(scaler_dir, 'y_scaler_' + ticker + '.pkl'), 'wb'))

    return x_train, y_train

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Please pass ticker"

    ticker = sys.argv[1]

    DATASET_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    SCALER_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scaler')

    df = get_stock_data(ticker, DATASET_DIR)

    x, y, _, _ = lstm_preprocess_data(df, SCALER_DIR)

    print(x.shape, y.shape)
    # (N, 50, 19) (N, 1)