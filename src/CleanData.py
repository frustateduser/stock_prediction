import pandas as pd

def cleanData(data: pd.DataFrame) -> pd.DataFrame:
    # Remove missing values
    data.dropna(inplace=True)

    # Convert data types
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Open'] = pd.to_numeric(data['Open'])
    data['High'] = pd.to_numeric(data['High'])
    data['Low'] = pd.to_numeric(data['Low'])
    data['Close'] = pd.to_numeric(data['Close'])
    data['Adj Close'] = pd.to_numeric(data['Adj Close'])
    data['Volume'] = pd.to_numeric(data['Volume'])

    # Remove unnecessary columns
    data.drop(['Adj Close'], axis=1, inplace=True)

    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()

    # Calculate moving averages
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    # Calculate exponential moving averages
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()

    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    up_days = delta.copy()
    up_days[delta <= 0] = 0
    down_days = abs(delta.copy())
    down_days[delta > 0] = 0
    RS_up = up_days.ewm(com=13, adjust=False).mean()
    RS_down = down_days.ewm(com=13, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + RS_up / RS_down))

    # Calculate Bollinger Bands
    data['Upper_BB'] = data['MA_20'] + 2*data['Close'].rolling(window=20).std()
    data['Lower_BB'] = data['MA_20'] - 2*data['Close'].rolling(window=20).std()

    # Remove rows with missing values
    data.dropna(inplace=True)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'MA_50', 'MA_200', 'EMA_50', 'EMA_200', 'MA_20', 'RSI', 'Upper_BB', 'Lower_BB']]

    return data

