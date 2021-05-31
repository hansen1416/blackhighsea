import os
# import sys

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from keras.optimizers import Adam

"""
I defined 3 LSTM models. 
1. 64x128x64x64
2. 64x64x64x64
3. 64x64x64

I end up using the third one.

Each LSTM layer followed by a dropout layer, with dropout rate 0.2, 0.1, 0.1

Finally two dense layer, 16x1, 
the output is the predicted close price of the next trading day.
"""

def new_model_rnn1():
        
    mod=Sequential()
    mod.add(LSTM(units = 64, return_sequences = True, input_shape = (50,13)))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 128, return_sequences = True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64, return_sequences = True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 16, activation='tanh')))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 1, activation='tanh')))
    mod.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), \
        metrics=['accuracy','mean_squared_error'])

    return mod

def new_model_rnn2():
        
    mod=Sequential()
    mod.add(LSTM(units = 64, return_sequences = True, input_shape = (50,13)))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64, return_sequences = True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64, return_sequences = True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 16, activation='tanh')))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 1, activation='tanh')))
    mod.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), \
        metrics=['accuracy','mean_squared_error'])

    return mod

def new_model_rnn3():
        
    mod=Sequential()
    mod.add(LSTM(units = 64, return_sequences = True, input_shape = (50,13)))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64, return_sequences = True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add(LSTM(units = 64))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 16, activation='tanh')))
    mod.add(BatchNormalization())

    mod.add((Dense(units = 1, activation='tanh')))
    mod.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), \
        metrics=['accuracy','mean_squared_error'])

    return mod

def get_model_path(model, checkpoint_dir, ticker):
    if model == 1:
        return os.path.join(checkpoint_dir, 'rnn_' + str(ticker) + '_1.h5')
    elif model == 2:
        return os.path.join(checkpoint_dir, 'rnn_' + str(ticker) + '_2.h5')
    elif model == 3:
        return os.path.join(checkpoint_dir, 'rnn_' + str(ticker) + '_3.h5')

def load_saved_model(model, checkpoint_dir, ticker):
    """
    model trained on 300 stocks, 64x128x64x64, input shape (50,13)
    """
    return tf.keras.models.load_model(get_model_path(model, checkpoint_dir, ticker))