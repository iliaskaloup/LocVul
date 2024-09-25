import os
import subprocess
import sys
import pandas as pd

def install_gdown():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])

def download_file(url, output):
    import gdown
    try:
        gdown.download(url, output, quiet=False)
        print(f"Downloaded {output} successfully.")
    except Exception as e:
        print(f"Error downloading {output}: {e}")


install_gdown()

# URLs for the datasets
urls = [
    "https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw",
    "https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ",
    "https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V"
]

# Corresponding output file names
files = ["train.csv", "val.csv", "test.csv"]

# Directory to save the files
data_dir = os.path.join("data")

# Create the directory if it does not exist
os.makedirs(data_dir, exist_ok=True)

# Download each file
for url, file in zip(urls, files):
    output_path = os.path.join(data_dir, file)
    download_file(url, output_path)

# Load the datasets and concatenate them
data_frames = []
for file in files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    print(len(df))
    data_frames.append(df)

# Concatenate all data frames
combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined dataset
combined_output_path = os.path.join(data_dir, 'dataset.csv')
combined_df.to_csv(combined_output_path, index=False)

for file in files:
    file_path = os.path.join(data_dir, file)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing {file_path}: {e}")



