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
        "--variables_moe",
        nargs="+",
        help="One or more variables to extract from an MoE layer"
    )
    parser.add_argument(
        "--variables_meta",
        nargs="+",
        help="One or more variables to extract from metadata"
    )
    parser.add_argument(
        "--paths",
        nargs="+",  # Allows one or more values
        help="One or more paths to obtain data from (e.g., file paths)."
    )
    parser.add_argument(
        "--metric",
        default="throughput",
        help="The metric want to create a csv for: (throughput, mttft, gpu-balance, expert-freq, timeline, basic)"
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
    parser.add_argument(
        "--output_name",
        default="out",
        type=str,
        help="Name of the output csv"
    )
    parser.add_argument(
        "--do",
        default="save",
        type=str,
        help="What you want to do with the result: (save, print)"
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
        elif args.metric == "timeline":
            _df = pd.read_csv(os.path.join(path, "0/e2e.csv"), index_col=0)
            _df["throughput (toks/s)"] = (meta["batch_size"] * meta["world_size"]) / _df["latency (s)"]
            if args.variables_meta:
                for var in args.variables_meta:
                    _df[var] = meta[var]
            _df = _df.drop(columns=["latency (s)"])
            df.append(_df)
        elif args.metric == "basic":
            _df = pd.DataFrame()
            if args.variables_moe:
                __df = pd.read_csv(os.path.join(path, "0/moe_layer-0.csv"), index_col=0)
                _df = pd.concat([_df, __df[args.variables_moe]], axis=1)
            if args.variables:
                __df = pd.read_csv(os.path.join(path, "0/e2e.csv"), index_col=0)
                _df = pd.concat([_df, __df[args.variables]], axis=1)
            if args.variables_meta:
                for var in args.variables_meta:
                    _df[var] = meta[var]
            if len(df) == 0:
                df = _df
            else:
                df = pd.concat([_df, df])
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
        elif args.metric == "imbalance":
            df = None
            for layer_idx in range(args.num_moe_layers):
                _df = None
                for rank_idx in range(args.world_size):
                    __df = pd.read_csv(os.path.join(path, f"{rank_idx}/moe_layer-{layer_idx}.csv"), index_col=0)
                    __df["expert distribution"] = __df["expert distribution"].apply(ast.literal_eval)
                    __df = pd.DataFrame(__df["expert distribution"].tolist(), index=__df.index)

                    num_gpus = args.world_size
                    cols_per_gpu = 16  # Number of columns assigned to each GPU

                    # Compute the sum for each GPU
                    gpu_sums = {
                        gpu_idx: __df.iloc[:, gpu_idx * cols_per_gpu : (gpu_idx + 1) * cols_per_gpu].sum(axis=1)
                        for gpu_idx in range(num_gpus)
                    }

                    # Combine the results into a new DataFrame
                    gpu_sums_df = pd.DataFrame(gpu_sums, index=__df.index)

                    if _df is None:
                        _df = gpu_sums_df
                    else:
                        _df += gpu_sums_df

                _df["max"] = _df.max(axis=1)
                _df["min"] = _df.min(axis=1)
                _df["imbalance (%)"] = ((_df["max"] - _df["min"]) / _df["min"]) * 100
                _df = _df["imbalance (%)"]
                if df is None:
                    df = _df
                else:
                    df += _df
            df /= args.num_moe_layers
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


    if isinstance(df, dict) or isinstance(df, list):
        if isinstance(df[0], pd.DataFrame):
            _min = min(df, key=lambda x: len(x)).shape[0]
            new_df = pd.DataFrame()
            for _df in df:
                new_df = pd.concat([new_df, _df[:_min]])
            df = new_df
        else:
            df = pd.DataFrame(df)

    return df


def main():
    args = parse_args()

    # Load and process data
    df = load_data(args)
    if args.do == "print":
        print(df)
    else:
        df.to_csv(f"../data_processed/{args.output_name}.csv")

if __name__ == "__main__":
    main()
