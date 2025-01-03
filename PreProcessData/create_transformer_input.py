import json
import re
import os

###############################################################################
# 1. Hard-Coded Configuration
###############################################################################

# Hard-coded paths to the normalized JSON files for each ticker.
# Adjust paths to match your actual folder structure and filenames.
JSON_FILES = {
    "VIX":       r"data_processed_imgs\^VIX\1d\regression_data\^VIX_1d_regression_data_normalized.json",
    "BTC-USD":   r"data_processed_imgs\BTC-USD\1d\regression_data\BTC-USD_1d_regression_data_normalized.json",
    "CL=F":      r"data_processed_imgs\CL=F\1d\regression_data\CL=F_1d_regression_data_normalized.json",
    "DIA":       r"data_processed_imgs\DIA\1d\regression_data\DIA_1d_regression_data_normalized.json",
    "EURUSD=X":  r"data_processed_imgs\EURUSD=X\1d\regression_data\EURUSD=X_1d_regression_data_normalized.json",
    "GLD":       r"data_processed_imgs\GLD\1d\regression_data\GLD_1d_regression_data_normalized.json",
    "HG=F":      r"data_processed_imgs\HG=F\1d\regression_data\HG=F_1d_regression_data_normalized.json",
    "IEF":       r"data_processed_imgs\IEF\1d\regression_data\IEF_1d_regression_data_normalized.json",
    "JPY=X":     r"data_processed_imgs\JPY=X\1d\regression_data\JPY=X_1d_regression_data_normalized.json",
    "QQQ":       r"data_processed_imgs\QQQ\1d\regression_data\QQQ_1d_regression_data_normalized.json",
    "SPY":       r"data_processed_imgs\SPY\1d\regression_data\SPY_1d_regression_data_normalized.json",
    "VNQ":       r"data_processed_imgs\VNQ\1d\regression_data\VNQ_1d_regression_data_normalized.json",
}

# Output file where the combined data will be saved.
OUTPUT_JSON = "all_tickers_combined.json"


###############################################################################
# 2. Helper Functions
###############################################################################

def extract_date_from_key(key: str) -> str:
    """
    Given a JSON key that typically looks like:
      'BTC-USD_1d_16c_2014-10-03 00-00-00.png'
    use a regex to parse out '2014-10-03'.

    Adjust the pattern if your filenames differ.
    """
    pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2}) .*\.png")
    match = pattern.match(key)
    if match:
        return match.group(1)  # e.g., "2014-10-03"
    return None


def load_json_and_extract_trends(json_path: str) -> dict:
    """
    Loads one JSON file of the form:
      {
        "BTC-USD_1d_16c_2014-10-03 00-00-00.png": {
          "shape": "...",
          "trend_strength": 71.87418515024254
        },
        ...
      }

    Returns a dict mapping:
      date_str -> trend_strength

    e.g., { '2014-10-03': 71.874185, '2014-10-04': 72.939504, ... }
    """
    date_to_value = {}
    if not os.path.isfile(json_path):
        print(f"[WARN] File not found: {json_path}")
        return date_to_value

    with open(json_path, "r") as f:
        data = json.load(f)

    for key, info in data.items():
        date_str = extract_date_from_key(key)
        if date_str is None:
            # Skip entries where date can't be parsed.
            continue
        # Extract trend_strength; default to 0.0 if missing.
        trend_val = info.get("trend_strength", 0.0)
        date_to_value[date_str] = trend_val

    return date_to_value


def build_aggregated_data(json_files: dict) -> (dict, list):
    """
    Given a dict { ticker: json_path }, load each file, parse date -> trend_strength,
    and aggregate into a structure:

      {
        date_str: {
          tickerA: val,
          tickerB: val,
          ...
        },
        ...
      }

    Also returns a list of ticker names in the order they were provided
    (or you can sort them, if you prefer).
    """
    date_dict = {}   # date_str -> { ticker -> val }
    tickers = list(json_files.keys())  # We keep the keys in insertion order

    for ticker, path in json_files.items():
        print(f"Loading data for {ticker} from {path}")
        date_to_val = load_json_and_extract_trends(path)

        for d_str, val in date_to_val.items():
            if d_str not in date_dict:
                date_dict[d_str] = {}
            date_dict[d_str][ticker] = val

    return date_dict, tickers


def convert_to_list(date_dict: dict, tickers: list) -> dict:
    """
    Convert from:
      {
        '2014-10-03': { 'VIX': 71.87, 'BTC-USD': 52.4, ... },
        '2014-10-04': { 'VIX': 72.93, 'BTC-USD': 54.1, ... }
      }
    to:
      {
        '2014-10-03': [71.87, 52.4, ...],
        '2014-10-04': [72.93, 54.1, ...]
      }

    The list order corresponds to the order of `tickers`.
    If a ticker is missing for a date, we default to 0.0.
    """
    final_data = {}

    for date_str, ticker_map in date_dict.items():
        row = []
        for t in tickers:
            val = ticker_map.get(t, 0.0)
            row.append(val)
        final_data[date_str] = row

    return final_data


def remove_dates_with_too_many_zeros(final_data, max_zeros=3):
    """
    Removes entries (dates) from `final_data` if
    they contain more than `max_zeros` zero-values.
    """
    cleaned_data = {}
    for date_str, values in final_data.items():
        zero_count = sum(1 for v in values if v == 0.0)
        if zero_count <= max_zeros:
            cleaned_data[date_str] = values
    return cleaned_data


###############################################################################
# 3. Main Script
###############################################################################

def main():
    # Step A: Aggregate all ticker data into date -> {ticker: val}
    date_dict, tickers = build_aggregated_data(JSON_FILES)

    # Step B: Convert that structure into date -> [val_ticker0, val_ticker1, ...]
    final_data = convert_to_list(date_dict, tickers)

    # 1. Remove bad records (dates with >3 zeros)
    final_data_cleaned = remove_dates_with_too_many_zeros(final_data, max_zeros=3)

    # 2. Sort by date (keys)
    final_data_sorted = {}
    for d_str in sorted(final_data_cleaned.keys()):
        final_data_sorted[d_str] = final_data_cleaned[d_str]

    # 3. Write to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "tickers_order": tickers,
            "data": final_data_sorted
        }, f, indent=2)

    print(f"\n[INFO] Wrote combined data to {OUTPUT_JSON}")
    print(f"[INFO] Tickers included: {tickers}")
    print(f"[INFO] Total dates found (after filtering): {len(final_data_sorted.keys())}")


if __name__ == "__main__":
    main()
