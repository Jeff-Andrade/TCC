import pandas as pd
from lightkurve import search_lightcurve
import os
from tqdm import tqdm  # Progress bar

# Load your dataset
df = pd.read_csv('../data/negative_data.csv')  # Replace with your actual file path

# Get unique TIC IDs
unique_tids = df['tid'].unique()

# Create output directory
output_dir = '../data/negative_data'
os.makedirs(output_dir, exist_ok=True)

# Get list of already downloaded files to skip them
downloaded_files = {f.split("_")[1].split(".")[0] for f in os.listdir(output_dir) if f.endswith('.fits')}

# Loop with progress bar
for tid in tqdm(unique_tids, desc="Downloading Light Curves", unit="TIC"):
    # Check if file already exists
    if str(tid) in downloaded_files:
        tqdm.write(f"Skipping TIC {tid} (already downloaded)")
        continue  # Skip to next TIC

    try:
        # Search and download
        lc_collection = search_lightcurve(f'TIC {tid}', mission='TESS')
        if lc_collection is not None and len(lc_collection) > 0:
            light_curve = lc_collection[0].download()

            if light_curve is not None:
                # Save to file
                filename = os.path.join(output_dir, f"TIC_{tid}.fits")
                light_curve.to_fits(filename, overwrite=True)
                tqdm.write(f"Saved light curve for TIC {tid} to {filename}")
            else:
                tqdm.write(f"No light curve data downloaded for TIC {tid}.")
        else:
            tqdm.write(f"No light curves found for TIC {tid}.")

    except Exception as e:
        tqdm.write(f"Error processing TIC {tid}: {e}")
