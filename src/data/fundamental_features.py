import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import pytz

def add_fundamental_features(stock_data_dict):
    """
    Add fundamental metrics to multiple stock DataFrames.
    
    Args:
        stock_data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values
    
    Returns:
        dict: Dictionary with enhanced DataFrames including fundamental features
    """
    enhanced_data = {}
    
    for ticker, df in stock_data_dict.items():
        logging.info(f"Adding fundamental features for {ticker}")
        try:
            # Create a copy to avoid modifying the original
            enhanced_df = df.copy()
            
            # Initialize fundamental columns with static values
            fundamental_metrics = {
                'PE_Ratio': None, 
                'Forward_PE': None, 
                'PEG_Ratio': None, 
                'Price_to_Book': None, 
                'EPS': None, 
                'Profit_Margin': None, 
                'ROE': None, 
                'ROA': None, 
                'Debt_to_Equity': None,
                'Current_Ratio': None, 
                'Quick_Ratio': None, 
                'Dividend_Yield': None,
                'Fundamental_Score': 0
            }
            
            # Add columns to dataframe
            for metric, default_value in fundamental_metrics.items():
                enhanced_df[metric] = default_value
                
            # Get the most recent fundamental data using yfinance
            stock = yf.Ticker(ticker)
            
            try:
                # Get current info from Yahoo Finance
                ratios = stock.info
                
                if ratios:
                    # Map yfinance info dictionary keys to our column names
                    info_mapping = {
                        'trailingPE': 'PE_Ratio',
                        'forwardPE': 'Forward_PE',
                        'pegRatio': 'PEG_Ratio',
                        'priceToBook': 'Price_to_Book',
                        'trailingEps': 'EPS',
                        'profitMargins': 'Profit_Margin',
                        'returnOnEquity': 'ROE',
                        'returnOnAssets': 'ROA',
                        'debtToEquity': 'Debt_to_Equity',
                        'currentRatio': 'Current_Ratio',
                        'quickRatio': 'Quick_Ratio',
                        'dividendYield': 'Dividend_Yield'
                    }
                    
                    # Fill the most recent values for all dates
                    for yf_key, df_column in info_mapping.items():
                        if yf_key in ratios and ratios[yf_key] is not None:
                            enhanced_df[df_column] = ratios[yf_key]
            except Exception as e:
                logging.warning(f"Error getting fundamental ratios for {ticker}: {e}")
            
            # Skip the quarterly data matching for now since it's causing timezone issues
            # We'll just use the most recent values instead
            
            # Calculate a fundamental score - simple version
            enhanced_df['Fundamental_Score'] = 0
            
            # Add to score for good values in positive metrics
            for pos_metric in ['ROE', 'ROA', 'Current_Ratio', 'Quick_Ratio']:
                if pos_metric in enhanced_df.columns:
                    # For ROE and ROA, values above 0.1 (10%) are good
                    if pos_metric in ['ROE', 'ROA']:
                        good_mask = enhanced_df[pos_metric] > 0.1
                    # For ratios, values above 1.5 are generally good
                    else:
                        good_mask = enhanced_df[pos_metric] > 1.5
                        
                    enhanced_df.loc[good_mask, 'Fundamental_Score'] += 1
            
            # Subtract from score for bad values in negative metrics
            for neg_metric in ['Debt_to_Equity']:
                if neg_metric in enhanced_df.columns:
                    # High debt-to-equity is a negative
                    high_debt_mask = enhanced_df[neg_metric] > 2.0
                    enhanced_df.loc[high_debt_mask, 'Fundamental_Score'] -= 1
            
            # Add P/E ratio evaluation
            if 'PE_Ratio' in enhanced_df.columns:
                # Very high P/E could be a warning sign (overvalued)
                high_pe_mask = enhanced_df['PE_Ratio'] > 50
                enhanced_df.loc[high_pe_mask, 'Fundamental_Score'] -= 1
                
                # Reasonable P/E could be a good sign (fairly valued)
                good_pe_mask = (enhanced_df['PE_Ratio'] > 10) & (enhanced_df['PE_Ratio'] < 25)
                enhanced_df.loc[good_pe_mask, 'Fundamental_Score'] += 1
            
            # Store the enhanced dataframe
            enhanced_data[ticker] = enhanced_df
            
        except Exception as e:
            logging.error(f"Error processing fundamental data for {ticker}: {e}")
            # If there's an error, return the original dataframe
            enhanced_data[ticker] = df
    
    return enhanced_data