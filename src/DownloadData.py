import yfinance as yf
import pandas as pd
import datetime


# Function to get the stock's first listing date on market
def stockListingDate(stockName):
    # Download the historical stock data
    stock = yf.Ticker(stockName)
    # Get the historical data for the stock (fetching the max period to get the earliest available data)
    hist_data = stock.history(period="max")
    
    # Check if historical data is available
    if not hist_data.empty:
        # Get the first available date in the historical data
        listing_date = hist_data.index[0].date()
        return str(listing_date)
    else:
        return f"No historical data available for {stockName}"
    

#returns todays date
def todayDate():
    return datetime.datetime.now().strftime('%Y-%m-%d')


# downloads stock data from Yahoo Finance using yfinance library
def downloadStockData(stock: str) -> pd.DataFrame:
    try:
        df = yf.download(stock, start=stockListingDate(stock), end=todayDate())
        if df.empty:
            raise ValueError(f"No data available for {stock}")
        return df
    except Exception as e:
        print(f"Error downloading data for {stock}: {e}")
        return pd.DataFrame()
