import pandas as pd

def add_moving_averages(df):
    """Add various moving averages to the dataframe"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages (gives more weight to recent prices)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    return df

def add_momentum_indicators(df):
    """Add momentum indicators to the dataframe"""
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Rate of Change
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    
    return df

def add_volatility_indicators(df):
    """Add volatility indicators to the dataframe"""
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    return df

def calculate_technical_indicators(stock_data_dict):
    """
    Calculate various technical indicators for a dictionary of stock dataframes.
    
    Args:
        stock_data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values
    
    Returns:
        dict: Dictionary with the same structure but with added technical indicators
    """
    enhanced_data = {}
    
    for ticker, df in stock_data_dict.items():
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Calculate various indicators
        enhanced_df = add_moving_averages(enhanced_df)
        enhanced_df = add_momentum_indicators(enhanced_df)
        enhanced_df = add_volatility_indicators(enhanced_df)
        
        # Store the enhanced dataframe
        enhanced_data[ticker] = enhanced_df
        
    return enhanced_data