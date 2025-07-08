"""
Converts CSV files to data.npy and mask.npy format for federated learning
"""

import os
import pandas as pd
import numpy as np

INPUT_FOLDER = "preprocessed"

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        filepath = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(filepath)

        # Convert -1 imputed values to mask
        data = df.to_numpy(dtype=np.float32)
        mask = (data != -1).astype(np.uint8)

        # Create output folder per file
        name = os.path.splitext(filename)[0]
        out_dir = os.path.join(INPUT_FOLDER, name)
        os.makedirs(out_dir, exist_ok=True)

        # Save data and mask arrays
        np.save(os.path.join(out_dir, "data.npy"), data)
        np.save(os.path.join(out_dir, "mask.npy"), mask)

        # Assign protocol based on filename
        protocol = "unknown"
        if "powerduck" in name.lower():
            protocol = "goose"
        elif "unsw" in name.lower():
            protocol = "http"

        with open(os.path.join(out_dir, "protocol.txt"), "w") as f:
            f.write(protocol)

        print(f"Processed {filename} → {name}/data.npy, mask.npy, protocol: {protocol}")
