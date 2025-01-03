import os
import pandas as pd

# Directory containing aligned data
data_dir = "macro_data/"

def check_missing_days():
    # Read all aligned CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith("_aligned.csv")]
    
    if not csv_files:
        print("No aligned CSV files found in the directory.")
        return
    
    # Dictionary to store dataframes and their date indices
    ticker_date_indices = {}
    
    for file in csv_files:
        ticker = file.split("_")[0]  # Extract ticker from filename
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        
        if data.empty:
            print(f"No data found in {file}.")
            continue
        
        ticker_date_indices[ticker] = data.index
        print(f"{ticker}: {len(data)} rows loaded, from {data.index.min()} to {data.index.max()}.")
    
    # Find the union and intersection of all date indices
    all_dates_union = pd.Index(sorted(set.union(*[set(idx) for idx in ticker_date_indices.values()])))
    all_dates_intersection = pd.Index(sorted(set.intersection(*[set(idx) for idx in ticker_date_indices.values()])))
    
    print("\n--- Date Range Analysis ---")
    print(f"Total dates in union: {len(all_dates_union)}")
    print(f"Total dates in intersection: {len(all_dates_intersection)}")
    
    # Check for missing dates for each ticker
    for ticker, dates in ticker_date_indices.items():
        missing_dates = all_dates_union.difference(dates)
        if not missing_dates.empty:
            print(f"{ticker} is missing {len(missing_dates)} days.")
            print(f"Missing dates: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")
        else:
            print(f"{ticker} has no missing days.")
    
    print("\n--- Consistency Check ---")
    if len(all_dates_union) == len(all_dates_intersection):
        print("All tickers have consistent date ranges.")
    else:
        print("Date ranges are inconsistent across tickers. Some have missing days.")

if __name__ == "__main__":
    check_missing_days()
