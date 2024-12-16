STOCK PRICE PREDICTION AND FORECASTING

This projet uses LSTM neural network to predict and forecast stock prices based on historical data.
The model is traine on daily closing prices and leverages past 60 days of data to predict future values

Key Features:
  Data Preprocessing: Scaled the closing prices using MinMaxScaler for efficient training.
  Sequence Creation: Used sliding window methodology to generate input-output pairs for the LSTM model.
  Model Training: Built and trained an LSTM model to predict stock prices.
  Forecasting: Extended predictions to forecast the next 30 days, starting from the last known sequence.
  Visualization: Includes plots for:
    Test set predictions versus actual prices.
    Forecasted prices alongside the historical sequence used for prediction.
