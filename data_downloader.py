import yfinance as yf
import os
import pandas as pd

# List of tickers representing the macroeconomic picture
tickers = [
    "DIA",        # Dow Jones Industrial Average ETF
    "QQQ",        # Nasdaq-100 ETF
    "SPY",        # S&P 500 ETF
    "GLD",        # Gold ETF
    "CL=F",       # WTI Crude Oil Futures
    "EURUSD=X",   # EUR/USD Exchange Rate
    "JPY=X",      # USD/JPY Exchange Rate
    "BTC-USD",    # Bitcoin
    "IEF",        # US 10-Year Treasury ETF
    "^VIX",       # Volatility Index
    "VNQ",        # Vanguard Real Estate ETF
    "HG=F"        # Copper Futures
]

# Directory to save data
output_dir = "macro_data/"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store data for each ticker
ticker_data = {}

# Function to fetch 1-day interval data for a ticker
def fetch_1d_data(ticker):
    try:
        print(f"Fetching 1-day data for {ticker}...")
        # Download 1-day interval data for the entire history
        data = yf.download(ticker, interval="1d", period="max", progress=False)
        
        if not data.empty:
            # Add to dictionary
            ticker_data[ticker] = data
            print(f"1-day data for {ticker} fetched successfully.")
        else:
            print(f"No 1-day data available for {ticker}.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Align data to match the ticker with the minimum history
def align_data():
    # Find the ticker with the shortest history
    min_start_date = max([data.index.min() for data in ticker_data.values()])
    min_end_date = min([data.index.max() for data in ticker_data.values()])
    print(f"Aligning data to range: {min_start_date} to {min_end_date}")
    
    # Truncate all tickers to this date range
    for ticker, data in ticker_data.items():
        aligned_data = data.loc[min_start_date:min_end_date]
        # Save aligned data to CSV
        csv_filename = os.path.join(output_dir, f"{ticker}_1d_data_aligned.csv")
        aligned_data.to_csv(csv_filename)
        print(f"Aligned data for {ticker} saved to {csv_filename}.")

# Main function to fetch and align data
def main():
    for ticker in tickers:
        fetch_1d_data(ticker)
    
    if ticker_data:
        align_data()

if __name__ == "__main__":
    main()
