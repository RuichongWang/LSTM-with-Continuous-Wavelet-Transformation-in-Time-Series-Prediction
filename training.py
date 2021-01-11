import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.chdir('YOUR_PATH')

train_data = np.load('train_res.npy', allow_pickle=True)
test_data = np.load('test_res.npy', allow_pickle=True)

X_train = list(map(lambda x: x[0], train_data))
y_train = list(map(lambda x: x[1], train_data))
X_test  = list(map(lambda x: x[0], test_data))
y_test  = list(map(lambda x: x[1], test_data))


def single_wave_trans(x, train_day=30, wavename='shan', max_scale=32):
    scales = range(1, max_scale + 1)
    [coefficients, frequencies] = pywt.cwt(x, scales, wavename)
    coefficients = np.swapaxes(coefficients, 0, 1)[-train_day:]
    coefficients = np.swapaxes(coefficients, 0, 1)
    return coefficients

X_train = list(map(lambda x: single_wave_trans(x), X_train))
X_test  = list(map(lambda x: single_wave_trans(x), X_test))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test  = np.array(X_test)
y_test  = np.array(y_test)

# LSTM model
epochs      = 150
batch_size  = 32

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=np.shape(X_train)[1:]))
model.add(
    tf.keras.layers.LSTM(units=32,
                         dropout=0.35,
                         activation='relu',
                         return_sequences=True))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='MSE', metrics=['mse'])
history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, y_test))


def evaluation(model, X_test, y_test):
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = np.reshape(predicted_stock_price,
                                       (np.shape(predicted_stock_price)[0]))
    errors = abs(predicted_stock_price - y_test)
    return [
        round(np.mean(errors), 5),
        round(mean_squared_error(y_test, predicted_stock_price), 5),
        round(mean_squared_error(y_test, predicted_stock_price)**(1 / 2), 5)
    ]


res = evaluation(model, X_test, y_test)
