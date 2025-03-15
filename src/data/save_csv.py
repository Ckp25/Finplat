import pandas as pd
import os
import logging
from datetime import datetime

def save_processed_data(processed_data_dict, output_dir="data/processed"):
    """
    Save each processed DataFrame to a separate CSV file.
    
    Args:
        processed_data_dict (dict): Dictionary with ticker symbols as keys and DataFrames as values
        output_dir (str): Directory to save the CSV files
    
    Returns:
        list: List of saved file paths
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # Go through each stock in the dictionary
    for ticker, df in processed_data_dict.items():
        # Create filename with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{ticker}_processed_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filepath)
        saved_files.append(filepath)
        logging.info(f"Saved processed {ticker} data to {filepath}")
    
    return saved_files