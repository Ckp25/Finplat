import requests
import logging
import numpy as np
from datetime import datetime
import pandas as pd

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

def add_sentiment_features(ticker, price_df):
    """
    Fetch sentiment data from Alpha Vantage and add it to price DataFrame
    with values persisting until new information arrives
    
    Args:
        ticker (str): Stock ticker symbol
        price_df (pd.DataFrame): DataFrame with price data
        api_key (str): Alpha Vantage API key
        
    Returns:
        pd.DataFrame: DataFrame with sentiment features added
    """
    api_key = read_api_key()
    
    enhanced_df = price_df.copy()
    
    # Initialize sentiment columns
    enhanced_df['Sentiment_Score'] = np.nan
    enhanced_df['News_Volume'] = 0
    enhanced_df['Sentiment_Relevance'] = np.nan
    
    # Fetch sentiment data from Alpha Vantage
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            logging.error(f"Failed to get data: Status code {response.status_code}")
            enhanced_df['Sentiment_Score'] = 0
            enhanced_df['Sentiment_Relevance'] = 0
            return enhanced_df
            
        sentiment_data = response.json()
        
        if "Information" in sentiment_data and "limit" in sentiment_data["Information"].lower():
            logging.warning(f"API limit reached: {sentiment_data['Information']}")
            enhanced_df['Sentiment_Score'] = 0
            enhanced_df['Sentiment_Relevance'] = 0
            return enhanced_df
            
        if "feed" not in sentiment_data or not sentiment_data["feed"]:
            logging.warning(f"No news data available for {ticker}")
            enhanced_df['Sentiment_Score'] = 0
            enhanced_df['Sentiment_Relevance'] = 0
            return enhanced_df
            
        # Track which days have news
        news_dates = []

        # Check if the DataFrame's index is timezone-aware
        is_tz_aware = enhanced_df.index.tzinfo is not None
        
        # Process each news item
        for article in sentiment_data["feed"]:
            time_published = article.get("time_published")
            if not time_published:
                continue
                
            # Parse article date (format: YYYYMMDDTHHMMSS)
            article_date = datetime.strptime(time_published[:8], "%Y%m%d")
            article_date = pd.Timestamp(article_date)

            # Make article_date timezone-aware if price_df has timezone info
            if is_tz_aware:
            # If price_df is timezone-aware, make article_date aware too
            # Use the same timezone as the DataFrame
                article_date = article_date.tz_localize(enhanced_df.index.tzinfo)
            
            # Get ticker-specific sentiment
            ticker_sentiment = None
            for sentiment in article.get("ticker_sentiment", []):
                if sentiment.get("ticker") == ticker:
                    ticker_sentiment = sentiment
                    break
            
            if not ticker_sentiment:
                continue
                
            sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
            relevance_score = float(ticker_sentiment.get("relevance_score", 0))
            
            # Find the next trading day
            next_trading_days = [d for d in enhanced_df.index if d >= article_date]
            
            if next_trading_days:
                trading_date = min(next_trading_days)
                news_dates.append(trading_date)
                
                # Update sentiment for this trading day
                if np.isnan(enhanced_df.loc[trading_date, 'Sentiment_Score']):
                    # First news of the day
                    enhanced_df.loc[trading_date, 'Sentiment_Score'] = sentiment_score
                    enhanced_df.loc[trading_date, 'Sentiment_Relevance'] = relevance_score
                    enhanced_df.loc[trading_date, 'News_Volume'] = 1
                else:
                    # Additional news, compute running average
                    current_volume = enhanced_df.loc[trading_date, 'News_Volume']
                    current_sentiment = enhanced_df.loc[trading_date, 'Sentiment_Score']
                    current_relevance = enhanced_df.loc[trading_date, 'Sentiment_Relevance']
                    
                    # Update with new weighted average
                    new_volume = current_volume + 1
                    new_sentiment = (current_sentiment * current_volume + sentiment_score) / new_volume
                    new_relevance = (current_relevance * current_volume + relevance_score) / new_volume
                    
                    enhanced_df.loc[trading_date, 'Sentiment_Score'] = new_sentiment
                    enhanced_df.loc[trading_date, 'Sentiment_Relevance'] = new_relevance
                    enhanced_df.loc[trading_date, 'News_Volume'] = new_volume
        
        # Sort news dates and forward fill values between news events
        news_dates = sorted(set(news_dates))
        
        if news_dates:
            # Handle period before first news
            if news_dates[0] > enhanced_df.index[0]:
                enhanced_df.loc[:news_dates[0], 'Sentiment_Score'] = 0
                enhanced_df.loc[:news_dates[0], 'Sentiment_Relevance'] = 0
            
            # Forward fill between each news event
            for i in range(len(news_dates)-1):
                current_date = news_dates[i]
                next_date = news_dates[i+1]
                
                # Get dates between
                between_dates = [d for d in enhanced_df.index if d > current_date and d < next_date]
                
                # Forward fill sentiment
                for date in between_dates:
                    enhanced_df.loc[date, 'Sentiment_Score'] = enhanced_df.loc[current_date, 'Sentiment_Score']
                    enhanced_df.loc[date, 'Sentiment_Relevance'] = enhanced_df.loc[current_date, 'Sentiment_Relevance']
            
            # Forward fill after last news
            last_news_date = news_dates[-1]
            after_dates = [d for d in enhanced_df.index if d > last_news_date]
            
            for date in after_dates:
                enhanced_df.loc[date, 'Sentiment_Score'] = enhanced_df.loc[last_news_date, 'Sentiment_Score']
                enhanced_df.loc[date, 'Sentiment_Relevance'] = enhanced_df.loc[last_news_date, 'Sentiment_Relevance']
        else:
            # No news dates found
            enhanced_df['Sentiment_Score'] = 0
            enhanced_df['Sentiment_Relevance'] = 0
        
        # Calculate weighted sentiment
        enhanced_df['Weighted_Sentiment'] = enhanced_df['Sentiment_Score'] * enhanced_df['Sentiment_Relevance']
        
        return enhanced_df
        
    except Exception as e:
        logging.error(f"Error in sentiment analysis for {ticker}: {e}")
        # Return default values in case of error
        enhanced_df['Sentiment_Score'] = 0
        enhanced_df['Sentiment_Relevance'] = 0
        return enhanced_df