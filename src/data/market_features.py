import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

def add_market_features(stock_data_dict):
    """
    Add market indices and commodity prices to multiple stock DataFrames.
    
    Args:
        stock_data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values
    
    Returns:
        dict: Dictionary with enhanced DataFrames including market features
    """
    enhanced_data = {}
    
    # Define market indices and commodities to track
    market_symbols = {
        'S&P500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DowJones': '^DJI',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
        'VIX': '^VIX'  # Volatility index, good for measuring market fear
    }
    
    # Get the earliest and latest dates across all stocks
    earliest_date = None
    latest_date = None
    # Also determine if any dataframes are timezone aware
    tz_info = None
    
    for df in stock_data_dict.values():
        if df is not None and not df.empty:
            if earliest_date is None or min(df.index) < earliest_date:
                earliest_date = min(df.index)
            if latest_date is None or max(df.index) > latest_date:
                latest_date = max(df.index)
            
            # Get timezone info from the first dataframe that has it
            if tz_info is None and df.index.tzinfo is not None:
                tz_info = df.index.tzinfo
    
    if earliest_date is None or latest_date is None:
        logging.error("No valid dates found in the stock data")
        return stock_data_dict
    
    # Fetch market data for the entire time range
    market_data = {}
    for name, symbol in market_symbols.items():
        try:
            logging.info(f"Fetching market data for {name} ({symbol})")
            # Add some padding to ensure we have data for the entire range
            market_df = yf.download(symbol, start=earliest_date - pd.Timedelta(days=5), 
                                    end=latest_date + pd.Timedelta(days=5),
                                    progress=False)
            
            if not market_df.empty:
                # Ensure consistent timezone handling
                if tz_info is not None and market_df.index.tzinfo is None:
                    # If our stock data is timezone aware but market data isn't
                    market_df.index = market_df.index.tz_localize('America/New_York')
                    logging.info(f"Localized {name} data to timezone")
                elif tz_info is None and market_df.index.tzinfo is not None:
                    # If our stock data is naive but market data is aware
                    market_df.index = market_df.index.tz_localize(None)
                    logging.info(f"Removed timezone info from {name} data")
                
                
                market_data[name] = market_df['Close']
                logging.info(f"Successfully fetched {len(market_df)} days of data for {name}")
            else:
                logging.warning(f"No data returned for {name} ({symbol})")
        except Exception as e:
            logging.error(f"Error fetching data for {name} ({symbol}): {e}")
    
    # Process each stock dataframe
    for ticker, df in stock_data_dict.items():
        logging.info(f"Adding market features for {ticker}")
        try:
            if df is None or df.empty:
                enhanced_data[ticker] = df
                continue
                
            # Create a copy to avoid modifying the original
            enhanced_df = df.copy()
            
            # Check if this dataframe is timezone aware
            is_tz_aware = enhanced_df.index.tzinfo is not None
            
            # Add market indices and commodity prices
            for name, series in market_data.items():
                # Ensure timezone compatibility for this specific dataframe
                aligned_series = series
                
                # Handle timezone differences if they exist
                if is_tz_aware and series.index.tzinfo is None:
                    # If this stock dataframe is timezone aware but market data isn't
                    aligned_series = series.copy()
                    aligned_series.index = aligned_series.index.tz_localize('America/New_York')
                elif not is_tz_aware and series.index.tzinfo is not None:
                    # If this stock dataframe is naive but market data is aware
                    aligned_series = series.copy()
                    aligned_series.index = aligned_series.index.tz_localize(None)
                
                # Create column name
                col_name = f"{name}_Price"
                
                # Align dates with stock dataframe
                aligned_series = aligned_series.reindex(enhanced_df.index, method='ffill')
                enhanced_df[col_name] = aligned_series
                
                # Calculate daily returns for the index/commodity
                returns_col = f"{name}_Return"
                enhanced_df[returns_col] = aligned_series.pct_change()
                
                # Calculate correlation between stock and index over rolling 30-day window
                stock_returns = enhanced_df['Close'].pct_change()
                enhanced_df[f"{name}_Correlation"] = stock_returns.rolling(30).corr(enhanced_df[returns_col])
                
                # Calculate relative strength (stock return minus market return)
                # Positive values mean the stock is outperforming the index
                enhanced_df[f"{name}_RelativeStrength"] = stock_returns - enhanced_df[returns_col]
                
                # Calculate beta (market sensitivity) over rolling 60-day window
                if name in ['S&P500', 'NASDAQ', 'DowJones']:
                    # Calculating beta only makes sense for stock indexes
                    cov = stock_returns.rolling(60).cov(enhanced_df[returns_col])
                    market_var = enhanced_df[returns_col].rolling(60).var()
                    # Avoid division by zero
                    enhanced_df[f"Beta_{name}"] = np.where(market_var > 0, cov / market_var, np.nan)
            
            # Add composite market indicator (average of all index correlations)
            index_correlation_cols = [col for col in enhanced_df.columns if 
                                      col.endswith('_Correlation') and 
                                      any(idx in col for idx in ['S&P500', 'NASDAQ', 'DowJones'])]
            
            if index_correlation_cols:
                enhanced_df['Avg_Market_Correlation'] = enhanced_df[index_correlation_cols].mean(axis=1)
            
            # Store the enhanced dataframe
            enhanced_data[ticker] = enhanced_df
            
        except Exception as e:
            logging.error(f"Error adding market features for {ticker}: {e}")
            # If there's an error, return the original dataframe
            enhanced_data[ticker] = df
    
    return enhanced_data