# Description: Main driver script for running the data pipeline.

import logging
from datetime import datetime
from collect import get_multiple_stocks_cached
from build_features import calculate_technical_indicators
from fundamental_features import add_fundamental_features
from market_features import add_market_features
from save_csv import save_processed_data
from sentiment_features import add_sentiment_features

def run_data_pipeline(tickers, period="1y", interval="1d", force_refresh=False):
    """
    Run the entire data pipeline from collection to feature engineering.
    
    Args:
        tickers (list): List of ticker symbols to process
        period (str): Time period to retrieve
        interval (str): Data interval
        force_refresh (bool): Force fresh API calls ignoring cache
        
    Returns:
        dict: Dictionary with processed data for each ticker
    """
    logging.info(f"Starting data pipeline for {len(tickers)} tickers")
    
    # Step 1: Collect raw data
    logging.info("Collecting raw stock data...")
    raw_stock_data = get_multiple_stocks_cached(tickers, period, interval, force_refresh)
    
    # Step 2: Calculate technical indicators
    logging.info("Calculating technical indicators...")
    enhanced_stock_data = calculate_technical_indicators(raw_stock_data)
    
    # Future steps would go here (e.g., feature selection, normalization)

    # Step 4: Add market features
    logging.info("Adding market features...")
    enhanced_stock_data = add_market_features(enhanced_stock_data)

    # Step 3: Add fundamental features
    logging.info("Adding fundamental features...")
    enhanced_stock_data = add_fundamental_features(enhanced_stock_data)

    # Step 5: Add sentiment features
    logging.info("Adding sentiment features...")
    for ticker, df in enhanced_stock_data.items():
        enhanced_stock_data[ticker] = add_sentiment_features(ticker, df)
    

    # Step 6: Save processed data
    logging.info("Saving processed data...")
    save_processed_data(enhanced_stock_data)
    
    logging.info("Data pipeline completed successfully")
    return enhanced_stock_data

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define tickers to process
    target_tickers = ["AAPL"]
    
    # Run the pipeline
    processed_data = run_data_pipeline(target_tickers, period="2y")
    
    # Print summary of processed data
    for ticker, df in processed_data.items():
        print(f"{ticker}: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print("-" * 50)