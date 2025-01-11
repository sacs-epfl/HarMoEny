import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os
import sys
import json
import argparse 
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Process variables and paths to create a pandas DataFrame.")
    parser.add_argument(
        "--variables",
        nargs="+",  # Allows one or more values
        help="One or more variables to extract from the runs (e.g., variable names)."
    )
    parser.add_argument(
        "--paths",
        nargs="+",  # Allows one or more values
        help="One or more paths to obtain data from (e.g., file paths)."
    )
    parser.add_argument(
        "--metric",
        default="throughput",
        help="The metric want to create a csv for: (throughput, mttft, gpu-balance, expert-freq)"
    )
    parser.add_argument(
        "--num_moe_layers",
        default=12,
        type=int,
        help="The number of moe layers in the model"
    )
    parser.add_argument(
        "--world_size",
        default=8,
        type=int,
        help="The number of GPUs or ranks"
    )
    return parser.parse_args()

def load_data(args):
    df = []
    for path in args.paths:
        with open(os.path.join(path, "data.json")) as f:
            meta = json.load(f)

        if args.metric == "throughput":
            row = {}
            for var in args.variables:
                row[var] = meta[var]
            _df = pd.read_csv(os.path.join(path, "0/e2e.csv"), index_col=0)
            row["throughput (toks/s)"] = meta["num_samples"] / _df.sum(axis=0)["latency (s)"]
            # There is no std div for throughput
            df.append(row)
        elif args.metric == "mttft":
            row = {}
            for var in args.variables:
                row[var] = meta[var]
            _df = pd.read_csv(os.path.join(path, "0/e2e.csv"), index_col=0)
            _df["TTFT (ms)"] = (_df["latency (s)"] * 1000) / (meta["batch_size"] * meta["world_size"])
            row["MTTFT (ms)"] = _df["TTFT (ms)"] .mean(axis=0)
            row["std div"] = _df["TTFT (ms)"] .std()
            row["num samples"] = len(_df["TTFT (ms)"] )
            df.append(row)
        elif args.metric == "gpu-balance":
            for layer_idx in range(args.num_moe_layers):
                _df = None
                for rank_idx in range(args.world_size):
                    __df = pd.read_csv(os.path.join(path, f"{rank_idx}/moe_layer-{layer_idx}.csv"), index_col=0)
                    __df = __df[["total number of tokens recv"]]
                    __df = __df.rename(columns={"total number of tokens recv": rank_idx})
                    if _df is None:
                        _df = __df
                    else:
                        _df = _df.merge(__df, on="iteration", how="outer")
                
                for itr in range(len(_df)):
                    row = {}

                    for var in args.variables:
                        row[var] = meta[var]

                    row["iteration"] = itr 
                    row["layer_idx"] = layer_idx
                    
                    df_row = _df.iloc[itr]
                    for i in df_row.index._data:
                        row[i] = df_row[i]

                    df.append(row)
        elif args.metric == "expert-freq":
            for layer_idx in range(args.num_moe_layers):
                _df = None
                for rank_idx in range(args.world_size):
                    __df = pd.read_csv(os.path.join(path, f"{rank_idx}/moe_layer-{layer_idx}.csv"), index_col=0)
                    __df["expert distribution"] = __df["expert distribution"].apply(ast.literal_eval)
                    __df = pd.DataFrame(__df["expert distribution"].tolist(), index=__df.index)

                    if _df is None:
                        _df = __df
                    else:
                        _df += __df

                for itr in range(len(_df)):
                    row = {}

                    for var in args.variables:
                        row[var] = meta[var]

                    row["iteration"] = itr 
                    row["layer_idx"] = layer_idx


                    df_row = _df.iloc[itr]
                    for i in df_row.index._data:
                        row[i] = df_row[i]

                    df.append(row)


    df = pd.DataFrame(df)
    return df


def main():
    args = parse_args()

    # Load and process data
    df = load_data(args)
    print("Combined DataFrame:")
    print(df)
    df.to_csv(f"../data_processed/{args.metric}.csv")

if __name__ == "__main__":
    main()