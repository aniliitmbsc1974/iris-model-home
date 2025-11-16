import pandas as pd
import numpy as np
import argparse
import os

def create_formatted_poison(file_path, poison_level=0.05, seed=42):
    """
    Loads the IRIS CSV file and injects realistic noise into a fraction of numeric rows,
    preserving format and range, and keeping 'species' unchanged.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - poison_level (float): Fraction of rows to poison (e.g., 0.05 for 5%).
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Poisoned DataFrame.
    """
    np.random.seed(seed)
    df = pd.read_csv(file_path)

    # Separate features and label
    features = df.drop(columns=["species"])
    labels = df["species"]

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    n_rows = len(features)
    n_poison = int(poison_level * n_rows)

    poisoned_indices = np.random.choice(n_rows, n_poison, replace=False)
    features_poisoned = features.copy()

    # Apply feature-wise Gaussian noise with rounding and clipping
    for col in numeric_cols:
        mean = features[col].mean()
        std = features[col].std()
        min_val = features[col].min()
        max_val = features[col].max()

        noise = np.random.normal(loc=0, scale=0.2 * std, size=n_poison)
        new_vals = features_poisoned.loc[poisoned_indices, col] + noise
        new_vals = np.clip(new_vals, min_val, max_val)
        new_vals = np.round(new_vals, 1)  # Match IRIS format: 1 decimal place

        features_poisoned.loc[poisoned_indices, col] = new_vals

    # Reattach the species column
    df_poisoned = features_poisoned.copy()
    df_poisoned["species"] = labels

    return df_poisoned

def main():
    parser = argparse.ArgumentParser(description="Poison IRIS dataset with realistic, formatted feature-wise noise.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("poison_level", type=float, help="Poisoning level (e.g., 0.05 for 5%)")

    args = parser.parse_args()
    input_file = args.input_file
    poison_level = args.poison_level

    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    if not (0 <= poison_level <= 1):
        print("Error: poison_level must be between 0 and 1 (e.g., 0.05 for 5%)")
        return

    df_poisoned = create_formatted_poison(input_file, poison_level)

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_poisoned_{int(poison_level * 100)}{ext}"
    df_poisoned.to_csv(output_file, index=False)

    print(f"Poisoned data saved to: {output_file}")

if __name__ == "__main__":
    main()

