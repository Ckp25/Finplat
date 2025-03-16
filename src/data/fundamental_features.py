import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime

def read_api_key(filepath=r"C:\Users\ckpra\OneDrive\Desktop\avapikey.txt"):
    """
    Read API key from a local text file
    
    Args:
        filepath (str): Path to the text file containing the API key
        
    Returns:
        str: The API key as a string
    """
    try:
        with open(filepath, 'r') as file:
            # Read the file and strip any whitespace
            api_key = file.read().strip()
        return api_key
    except FileNotFoundError:
        logging.error(f"API key file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading API key: {e}")
        return None

def add_fundamental_features(stock_data_dict):
    """
    Add fundamental metrics using Alpha Vantage API
    
    Args:
        stock_data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values
    
    Returns:
        dict: Dictionary with enhanced DataFrames including fundamental features
    """
    api_key = read_api_key()
    if not api_key:
        logging.error("No API key available for Alpha Vantage")
        return stock_data_dict
        
    enhanced_data = {}
    
    for ticker, df in stock_data_dict.items():
        logging.info(f"Adding fundamental features for {ticker}")
        
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Initialize fundamental columns
        enhanced_df['PE_Ratio'] = np.nan
        enhanced_df['EPS'] = np.nan
        enhanced_df['ROE'] = np.nan
        
        try:
            # 1. First, try to get quarterly earnings reports
            url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logging.error(f"Failed to get earnings data: Status code {response.status_code}")
            else:
                earnings_data = response.json()
                
                if "Information" in earnings_data and "limit" in earnings_data["Information"].lower():
                    logging.warning(f"API limit reached: {earnings_data['Information']}")
                elif "quarterlyEarnings" in earnings_data:
                    quarterly_earnings = earnings_data["quarterlyEarnings"]
                    
                    logging.info(f"Found {len(quarterly_earnings)} quarterly earnings reports for {ticker}")
                    
                    # Create a dict to store quarterly EPS values
                    eps_data = {}
                    
                    for quarter in quarterly_earnings:
                        fiscal_date = quarter.get("fiscalDateEnding")
                        reported_eps = quarter.get("reportedEPS")
                        
                        if fiscal_date and reported_eps:
                            try:
                                # Convert to datetime and numeric EPS
                                quarter_date = pd.to_datetime(fiscal_date)
                                eps_value = float(reported_eps)
                                
                                # Store in our dict
                                eps_data[quarter_date] = eps_value
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Error parsing earnings data: {e}")
                    
                    if eps_data:
                        # Convert to pandas Series and sort by date
                        eps_series = pd.Series(eps_data)
                        eps_series = eps_series.sort_index()

                        # Check the timezone of the dataframe's index
                        if enhanced_df.index.tz is not None:
                        # If the dataframe has a timezone, localize the eps_series dates to match
                            eps_series.index = eps_series.index.tz_localize(enhanced_df.index.tz)
                        else:
                        # If the dataframe doesn't have a timezone, ensure eps_series doesn't either
                            if eps_series.index.tz is not None:
                                eps_series.index = eps_series.index.tz_localize(None)

                        
                        # Forward fill to our daily dataframe
                        enhanced_df['EPS'] = eps_series.reindex(enhanced_df.index, method='ffill')
                        logging.info(f"Added EPS data for {ticker}")
                        
                        # 2. Get overview for more fundamental metrics
                        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            overview_data = response.json()
                            
                            # Check for PE ratio
                            if "TrailingPE" in overview_data:
                                try:
                                    pe_ratio = float(overview_data["TrailingPE"])
                                    # Create a time series of P/E ratio based on reported EPS and daily prices
                                    enhanced_df['PE_Ratio'] = enhanced_df['Close'] / enhanced_df['EPS']
                                    logging.info(f"Calculated P/E ratio from price and EPS for {ticker}")
                                except (ValueError, TypeError) as e:
                                    logging.warning(f"Error parsing P/E ratio: {e}")
                            
                            # Check for ROE
                            if "ReturnOnEquityTTM" in overview_data:
                                try:
                                    roe = float(overview_data["ReturnOnEquityTTM"])
                                    # For now, just use the current ROE value for all dates
                                    # In a real implementation, you'd want to get historical ROE data
                                    enhanced_df['ROE'] = roe
                                    logging.info(f"Added ROE data for {ticker}")
                                except (ValueError, TypeError) as e:
                                    logging.warning(f"Error parsing ROE: {e}")
                        
                            # 3. Try to get quarterly balance sheet data for historical ROE calculation
                            url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
                            response = requests.get(url)
                            
                            if response.status_code == 200:
                                balance_sheet_data = response.json()
                                
                                if "quarterlyReports" in balance_sheet_data:
                                    quarterly_reports = balance_sheet_data["quarterlyReports"]
                                    
                                    # Create dict to store quarterly total shareholder equity
                                    equity_data = {}
                                    
                                    for report in quarterly_reports:
                                        fiscal_date = report.get("fiscalDateEnding")
                                        total_equity = report.get("totalShareholderEquity")
                                        
                                        if fiscal_date and total_equity:
                                            try:
                                                # Convert to datetime and numeric value
                                                report_date = pd.to_datetime(fiscal_date)
                                                equity_value = float(total_equity)
                                                
                                                # Store in our dict
                                                equity_data[report_date] = equity_value
                                            except (ValueError, TypeError) as e:
                                                logging.warning(f"Error parsing balance sheet data: {e}")
                                    
                                    if equity_data:
                                        # Convert to pandas Series and sort by date
                                        equity_series = pd.Series(equity_data)
                                        equity_series = equity_series.sort_index()

                                        if enhanced_df.index.tz is not None:
                                            equity_series.index = equity_series.index.tz_localize(enhanced_df.index.tz)
                                        else:
                                            if equity_series.index.tz is not None:
                                                 equity_series.index = equity_series.index.tz_localize(None)
                                        
                                        # Calculate ROE for each quarter where we have both EPS and equity
                                        roe_data = {}
                                        
                                        for date in eps_series.index:
                                            if date in equity_series.index:
                                                eps = eps_series[date]
                                                equity = equity_series[date]
                                                
                                                if equity > 0:
                                                    # Annualize quarterly EPS (multiply by 4) and divide by equity
                                                    # per share (which is total equity / shares outstanding)
                                                    # As a simplification, assume shares outstanding is constant
                                                    roe_data[date] = (eps * 4) / (equity / 1e6)  # Rough approximation
                                        
                                        if roe_data:
                                            # Convert to pandas Series and sort by date
                                            roe_series = pd.Series(roe_data)
                                            roe_series = roe_series.sort_index()
                                            
                                            # Forward fill to our daily dataframe
                                            enhanced_df['ROE'] = roe_series.reindex(enhanced_df.index, method='ffill')
                                            logging.info(f"Added calculated ROE data for {ticker}")
                                    
        except Exception as e:
            logging.error(f"Error in fundamental analysis for {ticker}: {e}")
        
        # Store the enhanced dataframe
        enhanced_data[ticker] = enhanced_df
    
    return enhanced_data