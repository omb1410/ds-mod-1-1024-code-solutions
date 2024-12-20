# -*- coding: utf-8 -*-
"""time_series_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15rThLURFGAsMA2afUvw0eZFaacUUVQn1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.read_csv('/content/AirPassengers.csv')

df = pd.read_csv('AirPassengers.csv', index_col='Month')
df

df.index = pd.to_datetime(df.index)

df = df.sort_index()
df

df.plot()
plt.xlabel(""); #fat upward trend lol

from statsmodels.tsa.stattools import adfuller

adfuller(df)

df_diff1 = df.diff()
df_diff1.plot();

df_diff2 = df.diff(2)
df_diff2.plot();

df_log = np.log(df)
df_log.plot()
plt.xlabel("");

df_log_diff1 = df_log.diff()
df_log_diff1.plot()
plt.xlabel("");

adfuller(df_log_diff1.dropna())

df_log_diff2 = df_log.diff(2)
df_log_diff2.plot()
plt.xlabel("");

adfuller(df_log_diff2.dropna())

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_log_diff2.dropna());

plot_pacf(df_log_diff2.dropna());

df_log_diff12 = df_log.diff(12)
plot_acf(df_log_diff12.dropna(), lags=np.arange(1,30));

plot_pacf(df_log_diff12.dropna());

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

decomp = seasonal_decompose(df)

trend = decomp.trend
seasonal = decomp.seasonal
noise = decomp.resid

fig, ax = plt.subplots(nrows=4)
ax[0].plot(df)
ax[1].plot(trend)
ax[2].plot(seasonal)
ax[3].plot(noise)
plt.subplots_adjust(wspace=1, hspace=1)
plt.show()

noise.plot()

adfuller(noise.dropna())

plot_acf(noise.dropna(), lags=np.arange(1,30)); #q = [1, 2, 8]

q_vals = [1, 2, 8]

plot_pacf(noise.dropna(), lags=np.arange(1,30)); #p = [1, 10, 11]
plt.ylim(-0.7, 0.7)

p_vals = [1, 3, 6]

seasonal.plot()

adfuller(seasonal)

plot_acf(seasonal.dropna(), lags=np.arange(0,100));

plot_pacf(seasonal.dropna());

sp_vals = [1, 2, 8]

import itertools

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

best_rmse = float('inf')
best_model = None

train = df[:len(df)-50]

test = df[len(df)-50:]

for p in p_vals:
  for sp in sp_vals:
    print(f'Trying parameters: p={p}, q={q}, sp={sp}')
    model = SARIMAX(train, order=(p,0,0), seasonal_order=(sp,0,0,12)).fit()

    forecast = model.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean

    rmse = np.sqrt(mean_squared_error(test, forecast_values))

    if rmse < best_rmse:
      best_rmse = rmse
      best_model = model

best_rmse

df.describe()

df.plot()
best_model.fittedvalues.plot()
best_model.forecast(50).plot()

