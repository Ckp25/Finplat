import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Retrieve stock data for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period (str): Time period to retrieve (e.g., '1d', '5d', '1mo', '1y')
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        logging.error(f"Error retrieving data for {ticker}: {e}")
        return None

# Simple in-memory cache
_cache = {}

def get_stock_data_cached(ticker, period="1y", interval="1d", force_refresh=False):
    """
    Retrieve stock data with caching.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period
        interval (str): Data interval
        force_refresh (bool): Force a fresh API call ignoring cache
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    cache_key = f"{ticker}_{period}_{interval}"
    
    # Return cached data if available and not forcing refresh
    if not force_refresh and cache_key in _cache:
        return _cache[cache_key]
    
    # Get fresh data
    data = get_stock_data(ticker, period, interval)
    
    # Update cache
    if data is not None:
        _cache[cache_key] = data
    
    return data

def get_multiple_stocks_cached(tickers, period="1y", interval="1d", force_refresh=False):
    """
    Retrieve data for multiple ticker symbols with caching.
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period to retrieve
        interval (str): Data interval
        force_refresh (bool): Force fresh API calls ignoring cache
        
    Returns:
        dict: Dictionary with ticker symbols as keys and DataFrames as values
    """
    result = {}
    for ticker in tickers:
        result[ticker] = get_stock_data_cached(ticker, period, interval, force_refresh)
    return result

def save_to_csv(stock_prices, output_dir="data/raw"):
    """
    Save each stock DataFrame to a separate CSV file.
    
    Args:
        stock_prices (dict): Dictionary with ticker symbols as keys and DataFrames as values
        output_dir (str): Directory to save the CSV files
    
    Returns:
        list: List of saved file paths
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # Go through each stock in the dictionary
    for ticker in stock_prices.keys():
        # Get the DataFrame for this ticker
        stock_df = stock_prices[ticker]
        
        # Create filename
        filename = f"{ticker}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save the entire DataFrame to a CSV file
        stock_df.to_csv(filepath)
        saved_files.append(filepath)
        print(f"Saved {ticker} data to {filepath}")
    
    return saved_files


