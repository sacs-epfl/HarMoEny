from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/experiment_1/"
data = Data(directory_path)

num_ranks = 8
num_moe_layers = 12
num_experts = 32

def create_layer_latency_moe_vs_dense_vs_dense_param_matched():
    for moe_layer_idx in range(num_moe_layers):
        moe_df = data.load(f"deepspeed_policy/0/moe_layer-{moe_layer_idx}.csv")
        moe_df["type"] = "moe"
        dense_df = data.load(f"dense/layer-{2*moe_layer_idx}.csv")
        dense_df["type"] = "dense"
        dense_param_matched_df = data.load(f"dense_param_match/layer-{2*moe_layer_idx}.csv")
        dense_param_matched_df["type"] = "dense_param_matched"
        save_pd(pd.concat([moe_df[["type", "latency (ms)"]], dense_df[["type", "latency (ms)"]], dense_param_matched_df[["type", "latency (ms)"]]]), f"../data_processed/experiment_1/latency_layer-{moe_layer_idx}.csv")

def create_expert_token_distribution():
    for moe_layer_idx in range(num_moe_layers):
        df = None
        for rank in range(num_ranks):
            _df = data.load(f"deepspeed_policy/{rank}/moe_layer-{moe_layer_idx}.csv")
            _df = _df[["expert distribution"]]
            _df = _df.rename(columns={"expert distribution": rank})
            _df[rank] = _df[rank].apply(ast.literal_eval)
            if df is None:
                df = _df
            else:
                df = df.merge(_df, how="outer", on="iteration")
        df["expert token distribution"] = df.apply(lambda row: [sum(x) for x in zip(*row)], axis=1)
        df = df.drop(columns=range(num_ranks))
        for expert_idx in range(num_experts):
            df[expert_idx] = df["expert token distribution"].apply(lambda x: x[expert_idx])
        df = df.drop(columns=["expert token distribution"])
        df = df.div(df.sum(axis=1), axis=0)
        df = df * 100
        
        save_pd(df, f"../data_processed/experiment_1/expert_token_distribution_layer-{moe_layer_idx}.csv")

def create_working_to_idle_time():
    for moe_layer_idx in range(num_moe_layers):
        df = None
        for rank in range(num_ranks):
            _df = data.load(f"deepspeed_policy/{rank}/moe_layer-{moe_layer_idx}.csv")
            _df = _df[["comp latency (ms)"]]
            _df = _df.rename(columns={"comp latency (ms)": rank})
            if df is None:
                df = _df
            else:
                df = df.merge(_df, how="outer", on="iteration")
        cols = df.columns
        df["min (ms)"] = df[cols].min(axis=1)
        df["max (ms)"] = df[cols].max(axis=1)
        df["idle (ms)"] = df["max (ms)"] - df["min (ms)"]
        save_pd(df[["min (ms)", "idle (ms)"]], f"../data_processed/experiment_1/working_to_idle_time_layer-{moe_layer_idx}.csv")

if __name__ == "__main__":
    create_layer_latency_moe_vs_dense_vs_dense_param_matched()
    create_expert_token_distribution()
    create_working_to_idle_time()