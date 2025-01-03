import os

def save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder):
    """Save the candlestick image with a filename based on the ticker, timeframe, window size, and end date."""
    filename = f"{ticker}_{timeframe}_{window_size}c_{end_date}.png"
    filepath = os.path.join(output_folder, filename)
    image.save(filepath)
    #print(f"Saved: {filepath}")
    return filename