import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#Data Processing

df = pd.read_csv('AAPL_stock_data.csv')
df_close = df.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(df_close).reshape(-1, 1))
def create_sequence(df, seq_length):
    x, y = [], []
    for i in range(seq_length, len(df)):
        x.append(df[i - seq_length:i])
        y.append(df[i])
    return np.array(x), np.array(y)

x, y = create_sequence(scaled_data, 60)
train_size = int(len(x) * 0.8)
x_train, y_train, x_test, y_test = x[:train_size], y[:train_size], x[train_size:], y[train_size:]

#LSTM Model & graphs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences = True, input_shape =(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences = False),
    Dropout(0.25),
    Dense(25),
    Dense(1) # Predicting one value
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

model.fit(x_train, y_train, epochs =10, batch_size =32, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
print(y_pred.shape)
y_pred = scaler.inverse_transform(y_pred)

import matplotlib.pyplot as plt

plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test), color = 'blue', label = 'Actual')
plt.plot(df.index[-len(y_pred):], y_pred , color = 'red', label = 'Predicted')
plt.legend(loc = 'upper left')
plt.show()

#Forecast

def forecast(model, last_sequence, days, scaler):
    forecasted_prices = []
    seq = last_sequence.copy()
    for _ in range(days):
        pred = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]))
        forecasted_prices.append(pred[0,0])
        seq = np.append(seq[1:], pred, axis = 0)
    return scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))

future_prices = forecast(model, x_test[-1], 30, scaler)

plt.plot(range(df.index[-1] -len(x_test[-1]), df.index[-1]), scaler.inverse_transform(x_test[-1]), label ='Date', color = 'green')
plt.plot(range(df.index[-1], df.index[-1] + 30), future_prices, label = 'Forcasted', color = 'orange')
plt.legend(loc = 'upper left')
plt.show()