# %%
import yfinance as yf 
import pandas as pd
import numpy as np 
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# %%
ticker = yf.Ticker('MU')

# %%
data = ticker.history(period='1y', interval='1h')

# %%
data

# %%
data.isna().sum()

# %%
close = data[['Close']]
close

# %%
close.index = pd.to_datetime(close.index)

# %%
close = close.sort_index()

# %%
close.plot()

# %%
from statsmodels.tsa.stattools import adfuller

# %%
adfuller(close.dropna())

# %%
close_log = np.log(close)
close_log.plot()

# %%
close_diff1 = close.diff()
close_diff1.plot()

# %%
adfuller(close_diff1.dropna())

# %%
close_log_diff1 = close_log.diff()
close_log_diff1.plot()

# %%
adfuller(close_log_diff1.dropna())

# %%
decomp = seasonal_decompose(close, period=24)

# %%
trend = decomp.trend
seasonal = decomp.seasonal
noise = decomp.resid

# %%
fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(close)
ax[1].plot(trend)
ax[2].plot(seasonal)
ax[3].plot(noise)
plt.subplots_adjust(wspace=1, hspace=1)
plt.show()

# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# %%
plot_acf(close_diff1.dropna(), lags=np.arange(1, 30));
plt.ylim(-0.1, 0.1)

# %%
q_vals = [1, 2]

# %%
plot_pacf(close_diff1.dropna(), lags=np.arange(1, 30));
plt.ylim(-0.1, 0.1)

# %%
p_vals = [1, 2]
d_vals = [1]

# %%
monthly = close.resample('M').mean()
monthly.plot()

# %%
best_rmse = float('inf')
best_model = None

train = close[:len(close)-50]

test = close[len(close)-50:]

for p in p_vals:
  for d in d_vals:
    for q in q_vals:
      print(f'Trying parameters: p={p}, d={d}, q={q}')
      model = ARIMA(train, order=(p,d,q)).fit()

      forecast = model.get_forecast(steps=len(test))
      forecast_values = forecast.predicted_mean

      rmse = np.sqrt(mean_squared_error(test, forecast_values))

      if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

# %%
best_model.model_orders

# %%
best_model = ARIMA(data['Close'], order=(1,1,2)).fit()

# %%
best_rmse

# %%
close.plot()
best_model.fittedvalues.plot()
best_model.forecast(2).plot()

# %%
best_model.forecast(2)

# %%
len(best_model.fittedvalues)

# %%
len(data["Close"])

# %%
next_2_predictions = best_model.forecast(2)
plt.figure(figsize=(12, 6)) 
plt.plot(data['Close'][-7:], label='Original Data') 
plt.plot(best_model.fittedvalues[-7:], label='Fitted Values') 
plt.plot(pd.date_range(start=data.index[-1], periods=3, freq='H'), np.concatenate([np.array([data['Close'][-1]]), next_2_predictions]), label='Next 2 Predictions')
plt.xlabel('Date') 
plt.ylabel('Price') 
plt.title('Stock Price Prediction for Next 2 Timestamps') 
plt.legend()

# %%
np.concatenate([np.array([5]), next_2_predictions.values])

# %%
data['Close']

# %%
data.index[-1]

# %%
pd.date_range(start=data.index[-1], periods=3, freq='H')

# %%
np.array([data['Close'][-1]])

# %%



