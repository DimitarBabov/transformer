import os
import shutil
import json
import numpy as np
from sklearn.cluster import KMeans

def normalize_json(input_json_path):
    epsilon = 1e-6
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    base, ext = os.path.splitext(input_json_path)
    output_json_path = f"{base}_normalized{ext}"

    # Extract parameters
    max_dev_values = [item["max_dev"] for item in json_data.values() if isinstance(item, dict)]
    colored_pixels_ratios = [item["colored_pixels_ratio"] for item in json_data.values() if isinstance(item, dict)]
   
    max_dev_mean = np.mean(max_dev_values)
    colored_pixels_mean = np.mean(colored_pixels_ratios)
    

    modified_json = {
        "_comments": [
            "shape: sequence of slopes ('n' = negative, 'p' = positive).",          
            "trend_strength = (price_change / max_dev_norm)",
            "colored_pixels_ratio_norm: nomralized by the mean of the dataset."
        ]
    }

    for key, value in json_data.items():
        if not isinstance(value, dict):
            modified_json[key] = value
            continue

        max_dev = value["max_dev"]
        price_change = value["price_change"]
        colored_pixels_ratio = value["colored_pixels_ratio"]

        max_dev_norm = max_dev/max_dev_mean
        colored_pixels_ratio_norm = colored_pixels_ratio/colored_pixels_mean   

        trend_strength = price_change / max_dev_norm

        slopes = [value["slope_first"], value["slope_second"], value["slope_third"], value["slope_whole"]]
        shape = "".join("p" if s > 0 else "n" for s in slopes)

        modified_json[key] = {
            "shape": shape,
            "trend_strength": trend_strength/colored_pixels_ratio_norm,
        }

    with open(output_json_path, 'w') as file:
        json.dump(modified_json, file, indent=4)

    print(f"Normalized JSON saved to: {output_json_path}")
    return output_json_path