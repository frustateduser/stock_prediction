# stock prediction model using LSTM

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import yfinance as yfimport os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Improved visualization
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import Sequential
from keras.layers import Dense, LSTM

# Set environment to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def download_stock_data(stock: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads stock data from Yahoo Finance.
    
    Args:
        stock (str): The stock symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
    
    Returns:
        pd.DataFrame: The stock data.
    """
    try:
        df = yf.download(stock, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

def build_model(input_shape: tuple, lstm_units: int = 50, dense_units: int = 25) -> Sequential:
    """
    Builds the LSTM model.
    
    Args:
        input_shape (tuple): The shape of the input data.
        lstm_units (int): Number of LSTM units.
        dense_units (int): Number of dense units.
    
    Returns:
        Sequential: The compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(dense_units))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(df: pd.DataFrame, training_data_len: int, feature: str = 'Close'):
    """
    Prepares the data for training the LSTM model.
    
    Args:
        df (pd.DataFrame): The stock data DataFrame.
        training_data_len (int): The length of the training data.
        feature (str): The feature to predict.
    
    Returns:
        tuple: x_train, y_train, scaler, scaled_data, data
    """
    data = df.filter([feature])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train).reshape(-1, 60, 1)
    return x_train, np.array(y_train), scaler, scaled_data, data

def test_data_prep(scaled_data: np.ndarray, training_data_len: int):
    """
    Prepares the test data for the LSTM model.
    
    Args:
        scaled_data (np.ndarray): The scaled data array.
        training_data_len (int): The length of the training data.
    
    Returns:
        tuple: x_test, y_test
    """
    test_data = scaled_data[training_data_len - 60:]
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test).reshape(-1, 60, 1)
    y_test = scaled_data[training_data_len:]
    return x_test, y_test

def evaluate_model(y_test: np.ndarray, predictions: np.ndarray):
    """
    Evaluates the model's performance using RMSE, MAE, and MAPE.
    
    Args:
        y_test (np.ndarray): The true test values.
        predictions (np.ndarray): The predicted values.
    """
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
def plot_predictions(data: pd.DataFrame, training_data_len: int, predictions: np.ndarray):
    """
    Plots the predictions against the actual data.
    
    Args:
        data (pd.DataFrame): The original stock data.
        training_data_len (int): The length of the training data.
        predictions (np.ndarray): The predicted values.
    """
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model Predictions')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'], label='Train')
    plt.plot(valid[['Close', 'Predictions']], label='Actual vs Predictions')
    plt.legend(loc='lower right')
    plt.show()

def predict_next_day_price(scaler: MinMaxScaler, last_60_days: np.ndarray, model: Sequential):
    """
    Predicts the next day price based on the last 60 days of data.
    
    Args:
        scaler (MinMaxScaler): The scaler used to transform the data.
        last_60_days (np.ndarray): The last 60 days of closing prices.
        model (Sequential): The trained LSTM model.
    """
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(f"Predicted next day price: {pred_price[0][0]:.2f}")

def run_stock_prediction(stock: str, start_date: str, end_date: str, epochs: int = 10, batch_size: int = 32):
    """
    Runs the stock prediction workflow.
    
    Args:
        stock (str): The stock symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
    """
    # Download stock data
    df = download_stock_data(stock, start_date, end_date)
    if df.empty:
        print("No data available for the given stock.")
        return

    # Prepare training data
    training_data_len = math.ceil(len(df) * 0.8)
    x_train, y_train, scaler, scaled_data, data = prepare_data(df, training_data_len)

    # Build and train the model
    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Prepare testing data
    x_test, y_test = test_data_prep(scaled_data, training_data_len)

    # Predict stock prices
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate model performance
    evaluate_model(scaled_data[training_data_len:], predictions)

    # Plot the results
    plot_predictions(data, training_data_len, predictions)

    # Predict the next day price
    last_60_days = df['Close'].values[-60:]  # Use only the last 60 days of closing prices
    predict_next_day_price(scaler, last_60_days, model)

# Run stock prediction for AAPL
run_stock_prediction('AAPL', '2012-01-01', '2024-09-24', epochs=20, batch_size=50)

from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

# Get the stock quote
df = yf.download('AAPL',start='2012-01-01',end='2019-12-17')
print(df)

# Get the number of rows and columns in the data set
print(df.shape)

# Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .9)

print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()


# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the valid and predicted prices
print(valid)

# Get the quote
apple_quote = yf.download('AAPL', start='2012-01-01', end='2019-12-17')
# Create a new dataframe
new_df = apple_quote.filter(['Close'])
# Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append the past 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# Get the quote
apple_quote2 = yf.download('AAPL', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])