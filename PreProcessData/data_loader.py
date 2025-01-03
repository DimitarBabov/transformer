# File: data_loader.py
import pandas as pd

# Load financial data from CSV
def load_data(csv_file):
    """Load financial data from a CSV file."""
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'], utc= True)  # Ensure Date column is in datetime format
    df.set_index('Date', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


