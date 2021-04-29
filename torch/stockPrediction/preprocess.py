import os
import sys

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATASET_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

def get_stock_data(ticker, shape=13) -> pd.DataFrame:

    df = pd.read_csv(os.path.join(DATASET_DIR, ticker + '.csv'), \
        index_col='date', parse_dates=False)

    return df

def scale_data(df):

    df = df.drop('p_close', axis=1)

    print(df)

    x = df
    y = df[['close']].shift(-1)

    print(x.shape)

    x_scaler = MinMaxScaler(feature_range=(0,1)).fit(x)
    y_scaler = MinMaxScaler(feature_range=(0,1)).fit(y)

    features = x_scaler.transform(x)
    target = y_scaler.transform(y)

    print(features.shape)

    return features, target, x_scaler, y_scaler
