import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras import Sequential
from keras_tuner import RandomSearch
from keras.layers import Dense, LSTM, Input, Bidirectional, Conv1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from ta.trend import SMAIndicator as SMA, MACD
from ta.momentum import RSIIndicator as RSI, WilliamsRIndicator as WilliamsR
from ta.volatility import BollingerBands

# Set environment to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



# Download stock data
def download_stock_data(stock: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = yf.download(stock, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data available for {stock} between {start_date} and {end_date}")
        return df
    except Exception as e:
        print(f"Error downloading data for {stock}: {e}")
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
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(LSTM(lstm_units, return_sequences=False, dropout=0.2))
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
    # Calculate technical indicators
    df['SMA_50'] = SMA(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMA(df['Close'], window=200).sma_indicator()
    df['RSI'] = RSI(df['Close'], window=14).rsi()
    
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()

    df['MACD'] = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9).macd()
    df['WilliamsR'] = WilliamsR(df['High'], df['Low'], df['Close'], lookback=14).wr()

 

    data = df[[feature, 'SMA_50', 'SMA_200', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD', 'WilliamsR']]
    dataset = data.values
    scaler = StandardScaler()  # Or MinMaxScaler() left for the user to choose or experiment
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train).reshape(-1, 100, 1)
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
    test_data = scaled_data[training_data_len - 100:]
    x_test, y_test = [], []
    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i, 0])
    x_test = np.array(x_test).reshape(-1, 100, 1)
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
    r2 = r2_score(y_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R-squared: {r2:.2f}")
    
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
    plt.title('Model Predictions vs Actual Prices', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'], label='Training Data', color='blue', linestyle='solid', linewidth=2)
    plt.plot(valid['Close'], label='Actual Prices', color='green', linestyle='solid', linewidth=2)
    plt.plot(valid['Predictions'], label='Predicted Prices', color='red', linestyle='solid', linewidth=2)
    plt.legend(loc='lower right', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def predict_next_day_price(scaler: MinMaxScaler, prediction_days: np.ndarray, model: Sequential):
    """
    Predicts the next day price based on the last 60 days of data.
    
    Args:
        scaler (MinMaxScaler): The scaler used to transform the data.
        prediction_days (np.ndarray): The last days of closing prices.
        model (Sequential): The trained LSTM model.
    """
    prediction_days_scaled = scaler.transform(prediction_days.reshape(-1, 1))
    X_test = np.array([prediction_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print(f"Predicted next day price: {pred_price[0][0]:.2f}")

#returns todays date
def get_today_date():
    return datetime.datetime.now().strftime('%Y-%m-%d')



# Function to get the stock's first listing date on market
def get_stock_listing_date(ticker_symbol):
    """
    Function to get the stock listing date using yfinance.
    
    Parameters:
    ticker_symbol (str): The stock symbol (e.g., "AAPL" for Apple)
    
    Returns:
    str: The first date of trading.
    """
    # Download the historical stock data
    stock = yf.Ticker(ticker_symbol)
    # Get the historical data for the stock (fetching the max period to get the earliest available data)
    hist_data = stock.history(period="max")
    
    # Check if historical data is available
    if not hist_data.empty:
        # Get the first available date in the historical data
        listing_date = hist_data.index[0].date()
        return str(listing_date)
    else:
        return f"No historical data available for {ticker_symbol}"


# Renaming the hyperparameter tuning function to avoid conflict
def hyperparameter_build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

def run_stock_prediction(stock: str, epochs: int = 10, batch_size: int = 32):
    """
    Runs the stock prediction workflow.
    """
    # Download stock data
    df = download_stock_data(stock, get_stock_listing_date(stock), get_today_date())
    if df.empty:
        print("No data available for the given stock.")
        return

    # Prepare training data
    training_data_len = math.ceil(len(df) * 0.9)
    x_train, y_train, scaler, scaled_data, data = prepare_data(df, training_data_len)

    # Use the original build_model function to build the basic model
    model = build_model((x_train.shape[1], 1))

    # Hyperparameter tuning using Keras Tuner (now using the renamed function)
    tuner = RandomSearch(
        hyperparameter_build_model,  # Updated with the renamed function
        objective='mse',  # minimize the mean squared error
        max_trials=10,  # number of hyperparameter combinations to try increase for better results
        executions_per_trial=3,  # number of times to train each combination
        directory='my_dir',
        project_name='stock_prediction'
    )

    # Start tuning
    tuner.search(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Get the best model from hyperparameter tuning
    best_model = tuner.get_best_models()[0]

    # Train the final model with the best hyperparameters
    best_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Prepare testing data
    x_test, y_test = test_data_prep(scaled_data, training_data_len)

    # Predict stock prices
    predictions = best_model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate model performance
    evaluate_model(scaled_data[training_data_len:], predictions)

    # Plot the results
    plot_predictions(data, training_data_len, predictions)

    # Predict the next day price
    prediction_days = df['Close'].values[-366:]
    predict_next_day_price(scaler, prediction_days, best_model)


# run the stock prediction workflow for Apple (AAPL) stock
run_stock_prediction('AAPL', epochs=128, batch_size=64)
