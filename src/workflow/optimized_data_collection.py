import yfinance as yf
import pandas as pd
import numpy as np
import logging

def standardize_datetime_index(df):
    """
    Remove timezone information from dataframe index
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            # Remove timezone info
            df.index = df.index.tz_localize(None)
    return df

def collect_optimized_data(tickers, years=2, include_technicals=True, include_fundamentals=True, include_market=True):
    """
    Collect and process optimized stock data with selected important features.
    
    Args:
        tickers (list): List of ticker symbols to process
        years (int): Number of years of historical data to retrieve (default: 2)
        include_technicals (bool): Whether to include technical indicators
        include_fundamentals (bool): Whether to include fundamental features
        include_market (bool): Whether to include market-related features
        
    Returns:
        dict: Dictionary with ticker symbols as keys and DataFrames as values
    """
    # Convert years to period string for yfinance
    period = f"{years*365 + 252}d"
    
    # Step 1: Collect raw data
    raw_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                raw_data[ticker] = data
        except Exception as e:
            logging.error(f"Error retrieving data for {ticker}: {e}")
    
    for ticker in raw_data:
        raw_data[ticker] = standardize_datetime_index(raw_data[ticker])
    
    # Step 2: Add selected features
    enhanced_data = raw_data.copy()
    
    if include_technicals:
        enhanced_data = add_selected_technical_indicators(enhanced_data)
    
    if include_market:
        enhanced_data = add_selected_market_indicators(enhanced_data)
    
    if include_fundamentals:
        enhanced_data = add_selected_fundamental_indicators(enhanced_data)
    
    for ticker,df in enhanced_data.items():
        enhanced_data[ticker].drop(enhanced_data[ticker].head(252).index, inplace=True)

    
    return enhanced_data

def add_selected_technical_indicators(stock_data_dict):
    """Add 6 important technical indicators to each dataframe"""
    enhanced_data = {}
    
    for ticker, df in stock_data_dict.items():
        df_copy = df.copy()
        
        # 1. RSI (Relative Strength Index) - momentum oscillator
        delta = df_copy['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (Moving Average Convergence Divergence)
        df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
        df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        
        # 3. Bollinger Bands
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
        df_copy['BB_Std'] = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_Width'] = (df_copy['BB_Std'] * 4) / df_copy['BB_Middle']  # Normalized width
        
        # 4. Average True Range (ATR) - volatility indicator
        high_low = df_copy['High'] - df_copy['Low']
        high_close = (df_copy['High'] - df_copy['Close'].shift()).abs()
        low_close = (df_copy['Low'] - df_copy['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_copy['ATR_14'] = tr.rolling(window=14).mean()
        df_copy['ATR_Pct'] = df_copy['ATR_14'] / df_copy['Close'].shift(1) * 100  # Normalized ATR
        
        # 5. Rate of Change - momentum
        df_copy['ROC_10'] = df_copy['Close'].pct_change(periods=10) * 100
        
        # 6. Volume Indicator (On-Balance Volume)
        df_copy['OBV'] = (df_copy['Volume'] * 
                          ((df_copy['Close'] > df_copy['Close'].shift()).astype(int) - 
                           (df_copy['Close'] < df_copy['Close'].shift()).astype(int))).cumsum()
        
        enhanced_data[ticker] = df_copy
    
    
    return enhanced_data

def add_selected_market_indicators(stock_data_dict):
    """Add S&P 500, Gold, Oil, VIX data and Interest rates to each dataframe with detailed logging"""
    if not stock_data_dict:
        return stock_data_dict
    
    # Get date range from the first ticker's data
    first_ticker = list(stock_data_dict.keys())[0]
    start_date = stock_data_dict[first_ticker].index.min()
    end_date = stock_data_dict[first_ticker].index.max()
    
    print(f"Data range: {start_date} to {end_date}")
    
    enhanced_data = {}
    
    try:
        # Download market data - with verbose logging
        print("Downloading S&P 500 data...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
        sp500 = standardize_datetime_index(sp500)
        print(f"S&P 500 data shape: {sp500.shape}, non-null Close values: {sp500['Close'].count()}")
        
        print("Downloading Gold data...")
        gold = yf.download('GC=F', start=start_date, end=end_date)
        gold = standardize_datetime_index(gold)
        print(f"Gold data shape: {gold.shape}, non-null Close values: {gold['Close'].count()}")
        
        print("Downloading Oil data...")
        oil = yf.download('CL=F', start=start_date, end=end_date)
        oil = standardize_datetime_index(oil)
        print(f"Oil data shape: {oil.shape}, non-null Close values: {oil['Close'].count()}")
        
        print("Downloading VIX data...")
        vix = yf.download('^VIX', start=start_date, end=end_date)
        vix = standardize_datetime_index(vix)
        print(f"VIX data shape: {vix.shape}, non-null Close values: {vix['Close'].count()}")
        
        print("Downloading Treasury data...")
        treasury = yf.download('^TNX', start=start_date, end=end_date)
        treasury = standardize_datetime_index(treasury)
        print(f"Treasury data shape: {treasury.shape}, non-null Close values: {treasury['Close'].count()}")
        
        for ticker, df in stock_data_dict.items():
            print(f"Processing {ticker}...")
            df_copy = df.copy()
            
            # Check frequency of data
            print(f"Stock data frequency: {df.index.to_series().diff().median()}")
            print(f"Market data frequency: {sp500.index.to_series().diff().median()}")
            
            # Verify index types
            print(f"Stock index type: {type(df.index)}")
            print(f"Market index type: {type(sp500.index)}")
            
            # 1. Ensure indices are compatible (both DatetimeIndex)
            if not isinstance(df.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            
            # 2. Use reindex method for safer alignment
            # S&P 500
            if not sp500.empty:
                sp500_close = sp500['Close'].copy()
                df_copy['SP500_Price'] = sp500_close.reindex(df_copy.index, method='ffill')
                df_copy['SP500_Return'] = df_copy['SP500_Price'].pct_change()
                
                # Check data alignment for debugging
                print(f"SP500_Price non-null values: {df_copy['SP500_Price'].count()} out of {len(df_copy)}")
                
                # Calculate Beta more carefully
                if len(df_copy) > 60:
                    # Use only dates where both values exist
                    stock_returns = df_copy['Close'].pct_change().dropna()
                    market_returns = df_copy['SP500_Return'].dropna()
                    
                    # Find common dates
                    common_dates = stock_returns.index.intersection(market_returns.index)
                    if len(common_dates) > 60:
                        stock_returns = stock_returns.loc[common_dates]
                        market_returns = market_returns.loc[common_dates]
                        
                        # Rolling windows
                        windows = min(60, len(common_dates)-1)
                        beta_values = []
                        beta_dates = []
                        for i in range(windows, len(common_dates)):
                            window_stock = stock_returns.iloc[i-windows:i]
                            window_market = market_returns.iloc[i-windows:i]

                            current_date = common_dates[i]
                            beta_dates.append(current_date)
                            
                            if window_market.var() != 0:  # Avoid division by zero
                                beta = window_stock.cov(window_market) / window_market.var()
                                beta_values.append(beta)
                            else:
                                beta_values.append(np.nan)  # Using np.nan for consistency
                        
                        # Create Series with appropriate dates
                        beta_series = pd.Series(beta_values, index=beta_dates)
                        df_copy.loc[beta_dates, 'Beta_SP500'] = beta_series

            # Gold
            if not gold.empty:
                gold_close = gold['Close'].copy()
                df_copy['Gold_Price'] = gold_close.reindex(df_copy.index, method='ffill')
                df_copy['Gold_Return'] = df_copy['Gold_Price'].pct_change()
                
                # Check data alignment
                print(f"Gold_Price non-null values: {df_copy['Gold_Price'].count()} out of {len(df_copy)}")
                
                # Proceed with correlation only if we have enough data
                if df_copy['Gold_Return'].count() > 30:
                    stock_returns = df_copy['Close'].pct_change().dropna()
                    gold_returns = df_copy['Gold_Return'].dropna()
                    
                    common_dates = stock_returns.index.intersection(gold_returns.index)
                    
                    if len(common_dates) > 30:
                        windows = min(30, len(common_dates)-1)
                        corr_values = []
                        corr_dates = []
                        
                        for i in range(windows, len(common_dates)):
                            window_stock = stock_returns.loc[common_dates].iloc[i-windows:i]
                            window_gold = gold_returns.loc[common_dates].iloc[i-windows:i]

                            current_date = common_dates[i]
                            corr_dates.append(current_date)
                            
                            corr = window_stock.corr(window_gold)
                            corr_values.append(corr)
                        
                        # Create series directly with our collected dates
                        corr_series = pd.Series(corr_values, index=corr_dates)
                        df_copy.loc[corr_dates, 'Gold_Correlation'] = corr_series

            # Oil - FIXED FROM USING GOLD VARIABLES
            if not oil.empty:
                oil_close = oil['Close'].copy()
                df_copy['Oil_Price'] = oil_close.reindex(df_copy.index, method='ffill')  # Fixed from using gold_close
                df_copy['Oil_Return'] = df_copy['Oil_Price'].pct_change()

                if df_copy['Oil_Return'].count() > 30:
                    stock_returns = df_copy['Close'].pct_change().dropna()
                    oil_returns = df_copy['Oil_Return'].dropna()  # Fixed from using gold_returns
                    
                    common_dates = stock_returns.index.intersection(oil_returns.index)  # Need to recalculate for oil

                    if len(common_dates) > 30:
                        windows = min(30, len(common_dates)-1)
                        corr_values = []
                        corr_dates = []
                            
                        for i in range(windows, len(common_dates)):
                             window_stock = stock_returns.loc[common_dates].iloc[i-windows:i]
                             window_oil = oil_returns.loc[common_dates].iloc[i-windows:i]  # Fixed from window_gold

                             current_date = common_dates[i]
                             corr_dates.append(current_date)
                                
                             corr = window_stock.corr(window_oil)  # Fixed correlation
                             corr_values.append(corr)
                            
                        corr_series = pd.Series(corr_values, index=corr_dates)
                        df_copy.loc[corr_dates, 'Oil_Correlation'] = corr_series

            # VIX - ADDED SECTION
            if not vix.empty:
                vix_close = vix['Close'].copy()
                df_copy['VIX_Price'] = vix_close.reindex(df_copy.index, method='ffill')
                df_copy['VIX_Return'] = df_copy['VIX_Price'].pct_change()
                
                # Check data alignment
                print(f"VIX_Price non-null values: {df_copy['VIX_Price'].count()} out of {len(df_copy)}")
                
                if df_copy['VIX_Return'].count() > 30:
                    stock_returns = df_copy['Close'].pct_change().dropna()
                    vix_returns = df_copy['VIX_Return'].dropna()
                    
                    common_dates = stock_returns.index.intersection(vix_returns.index)
                    
                    if len(common_dates) > 30:
                        windows = min(30, len(common_dates)-1)
                        corr_values = []
                        corr_dates = []
                        
                        for i in range(windows, len(common_dates)):
                            window_stock = stock_returns.loc[common_dates].iloc[i-windows:i]
                            window_vix = vix_returns.loc[common_dates].iloc[i-windows:i]

                            current_date = common_dates[i]
                            corr_dates.append(current_date)
                            
                            corr = window_stock.corr(window_vix)
                            corr_values.append(corr)
                        
                        corr_series = pd.Series(corr_values, index=corr_dates)
                        df_copy.loc[corr_dates, 'VIX_Correlation'] = corr_series

            # Treasury - ADDED SECTION
            if not treasury.empty:
                treasury_close = treasury['Close'].copy()
                df_copy['Treasury_Price'] = treasury_close.reindex(df_copy.index, method='ffill')
                df_copy['Treasury_Return'] = df_copy['Treasury_Price'].pct_change()
                
                # Check data alignment
                print(f"Treasury_Price non-null values: {df_copy['Treasury_Price'].count()} out of {len(df_copy)}")
                
                if df_copy['Treasury_Return'].count() > 30:
                    stock_returns = df_copy['Close'].pct_change().dropna()
                    treasury_returns = df_copy['Treasury_Return'].dropna()
                    
                    common_dates = stock_returns.index.intersection(treasury_returns.index)
                    
                    if len(common_dates) > 30:
                        windows = min(30, len(common_dates)-1)
                        corr_values = []
                        corr_dates = []
                        
                        for i in range(windows, len(common_dates)):
                            window_stock = stock_returns.loc[common_dates].iloc[i-windows:i]
                            window_treasury = treasury_returns.loc[common_dates].iloc[i-windows:i]

                            current_date = common_dates[i]
                            corr_dates.append(current_date)
                            
                            corr = window_stock.corr(window_treasury)
                            corr_values.append(corr)
                        
                        corr_series = pd.Series(corr_values, index=corr_dates)
                        df_copy.loc[corr_dates, 'Treasury_Correlation'] = corr_series
            
            enhanced_data[ticker] = df_copy
            
            # Final check
            print(f"Final check for {ticker}:")
            for col in ['SP500_Price', 'Gold_Price', 'Oil_Price', 'VIX_Price', 'Treasury_Price', 
                        'Beta_SP500', 'Gold_Correlation', 'Oil_Correlation', 'VIX_Correlation', 'Treasury_Correlation']:
                if col in df_copy.columns:
                    print(f"  - {col}: {df_copy[col].count()} non-null values")
        
        column_mapping = {
        'Treasury_Price': 'Interest_Rate',
        'Treasury_Return': 'Interest_Rate_Change',
        'Treasury_Correlation': 'Interest_Rate_Correlation'
    }
        
        for ticker, df in enhanced_data.items():
            enhanced_data[ticker] = df.rename(columns=column_mapping)
        
        return enhanced_data
        
    except Exception as e:
        print(f"Error fetching market data: {e}")
        import traceback
        traceback.print_exc()
        # Return original data if market data fails
        return stock_data_dict
    
def add_selected_fundamental_indicators(stock_data_dict):
    """Add 3 important fundamental indicators to each dataframe"""
    enhanced_data = {}
    
    for ticker, df in stock_data_dict.items():
        df_copy = df.copy()
        
        try:
            # Get company fundamentals
            stock = yf.Ticker(ticker)
            #stock = standardize_datetime_index(stock)
            
            # 1. P/E Ratio
            quarterly_financials = stock.quarterly_financials
            if not quarterly_financials.empty:
                # Get quarterly EPS
                eps_data = {}
                
                for date in quarterly_financials.columns:
                    if 'BasicEPS' in quarterly_financials.index:
                        eps = quarterly_financials.loc['BasicEPS', date]
                        eps_data[date] = eps
                
                if eps_data:
                    eps_series = pd.Series(eps_data).sort_index()
                    # Forward fill EPS to daily data
                    df_copy['EPS'] = eps_series.reindex(df_copy.index, method='ffill')
                    # Calculate P/E ratio
                    df_copy['PE_Ratio'] = df_copy['Close'] / df_copy['EPS']
            
            # 2. Price to Book Ratio
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty and 'TotalAssets' in balance_sheet.index and 'TotalLiabilities' in balance_sheet.index:
                book_value_data = {}
                
                for date in balance_sheet.columns:
                    total_assets = balance_sheet.loc['TotalAssets', date]
                    total_liabilities = balance_sheet.loc['TotalLiabilities', date]
                    book_value = total_assets - total_liabilities
                    book_value_data[date] = book_value
                
                if book_value_data:
                    book_value_series = pd.Series(book_value_data).sort_index()
                    # Get shares outstanding
                    shares = stock.info.get('sharesOutstanding', None)
                    if shares:
                        # Calculate book value per share
                        book_value_per_share = book_value_series / shares
                        # Forward fill to daily data
                        df_copy['BookValuePerShare'] = book_value_per_share.reindex(df_copy.index, method='ffill')
                        # Calculate P/B ratio
                        df_copy['PB_Ratio'] = df_copy['Close'] / df_copy['BookValuePerShare']
            
            # 3. Dividend Yield (using historical dividends already in yfinance data)
            if 'Dividends' in df_copy.columns:
                # Calculate trailing 12-month dividends
                df_copy['TTM_Dividends'] = df_copy['Dividends'].rolling(252).sum()
                df_copy['Dividend_Yield'] = df_copy['TTM_Dividends'] / df_copy['Close'] * 100
        
        except Exception as e:
            logging.error(f"Error fetching fundamental data for {ticker}: {e}")
        
        enhanced_data[ticker] = df_copy
    
    return enhanced_data


ticker = 'MSFT'
data = collect_optimized_data([ticker],2)[ticker]
print(data.info())
print( data.index.inferred_type == "datetime64")
data.to_csv(f'D:\Finplat\data\processed\{ticker}_optimal.csv')