import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os
import sys
import json
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="Process variables and paths to create a pandas DataFrame.")
    parser.add_argument(
        '--variables',
        nargs='+',  # Allows one or more values
        help="One or more variables to extract from the runs (e.g., variable names)."
    )
    parser.add_argument(
        '--paths',
        nargs='+',  # Allows one or more values
        help="One or more paths to obtain data from (e.g., file paths)."
    )
    return parser.parse_args()

def load_data(paths, variables):
    df = []
    for path in paths:
        _df = pd.read_csv(os.path.join(path, "0/e2e.csv"), index_col=0)
        with open(os.path.join(path, "data.json")) as f:
            meta = json.load(f)
        row = {
            "throughput (toks/s)": meta["num_samples"] / _df.sum(axis=0)["latency (s)"]
        }
        for var in variables:
            row[var] = meta[var]

        df.append(row)
    df = pd.DataFrame(df)
    return df


def main():
    args = parse_args()
    variables = args.variables
    paths = args.paths

    print(f"Variables: {variables}")
    print(f"Paths: {paths}")

    # Load and process data
    dataframe = load_data(paths, variables)
    print("Combined DataFrame:")
    print(dataframe)

if __name__ == "__main__":
    main()