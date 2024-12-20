# -*- coding: utf-8 -*-
"""time_series_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GK1J0WUxVZiEW50WTa8kyhwLIlxw4aNH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/AirPassengers.csv')

df

df.info()

df['Month'] = pd.to_datetime(df['Month'])

df.set_index('Month', inplace=True)

df

df.plot()
plt.xlabel("");

rolling_21 = df.rolling(21).mean()
rolling_21.plot()
plt.xlabel("");

df['rolling_21'] = rolling_21
df['rolling_21_std'] = df['#Passengers'].rolling(21).std()
df.plot()
plt.xlabel("");
plt.title("Rolling Mean and Standard Deviation");
#The data is not stationary

df['diff_1'] = df['#Passengers'].diff(1)

df

df['diff_1'].plot()

rolling_7_diff1 = df['diff_1'].rolling(7).mean()
rolling_7_std_diff1 = df['diff_1'].rolling(7).std()
plt.plot(df.index, df['diff_1'])
plt.plot(df.index, rolling_7_diff1, color = 'orange')
plt.plot(df.index, rolling_7_std_diff1, color = 'green')
plt.xlabel("");
plt.title("Differences 1");

df['diff_2'] = df['#Passengers'].diff(2)

df

df['diff_2'].plot()

rolling_7_diff2 = df['diff_2'].rolling(7).mean()
rolling_7_std_diff2 = df['diff_2'].rolling(7).std()
plt.plot(df.index, df['diff_2'])
plt.plot(df.index, rolling_7_diff2, color = 'orange')
plt.plot(df.index, rolling_7_std_diff2, color = 'green')
plt.xlabel("");
plt.title("Differences 2");

df['diff_3'] = df['#Passengers'].diff(3)

df

df['diff_3'].plot()

rolling_7_diff3 = df['diff_3'].rolling(7).mean()
rolling_7_std_diff3 = df['diff_3'].rolling(7).std()
plt.plot(df.index, df['diff_3'])
plt.plot(df.index, rolling_7_diff3, color = 'orange')
plt.plot(df.index, rolling_7_std_diff3, color = 'green')
plt.xlabel("");
plt.title("Differences 3");

df['log_Passengers'] = np.log(df['#Passengers'])
df['log_Passengers'].plot()

df['log_diff1'] = df['log_Passengers'].diff(1)

log_rolling_14_diff1 = df['log_diff1'].rolling(14).mean()
log_rolling_14_std_diff1 = df['log_diff1'].rolling(14).std()
plt.plot(df.index, df['log_diff1'])
plt.plot(df.index, log_rolling_14_diff1, color = 'orange')
plt.plot(df.index, log_rolling_14_std_diff1, color = 'green')
plt.xlabel("");
plt.title("Log Differences 1");

df['log_diff2'] = df['log_Passengers'].diff(2)

log_rolling_14_diff2 = df['log_diff2'].rolling(14).mean()
log_rolling_14_std_diff2 = df['log_diff2'].rolling(14).std()
plt.plot(df.index, df['log_diff2'])
plt.plot(df.index, log_rolling_14_diff2, color = 'orange')
plt.plot(df.index, log_rolling_14_std_diff2, color = 'green')
plt.xlabel("");
plt.title("Log Differences 2");

df['log_diff3'] = df['log_Passengers'].diff(3)

log_rolling_7_diff3 = df['log_diff3'].rolling(7).mean()
log_rolling_7_std_diff3 = df['log_diff3'].rolling(7).std()
plt.plot(df.index, df['log_diff3'])
plt.plot(df.index, log_rolling_7_diff3, color = 'orange')
plt.plot(df.index, log_rolling_7_std_diff3, color = 'green')
plt.xlabel("");
plt.title("Log Differences 3");

from statsmodels.tsa.stattools import adfuller

results1 = adfuller(df['log_diff1'].dropna())
results1

result2 = adfuller(df['log_diff2'].dropna())
result2

result3 = adfuller(df['log_diff3'].dropna())
result3
