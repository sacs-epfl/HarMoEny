from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/systems/router_dynamic_skew"
data = Data(directory_path)

systems = ["deepspeed", "fastmoe", "fastermoe", "exflow", "harmony"]

def create_workload_throughput_iters_system():
    df = None
    for sys in systems:
        _df = data.load(f"{sys}/0/e2e.csv")
        meta = data.read_meta(sys)
        _df["throughput"] = meta["batch_size"] / _df["latency (s)"]
        _df = _df.rename(columns={"throughput": sys})
        if df is None:
            df = _df[[sys]]
        else:
            df = df.join(_df[[sys]], on="iteration", how="outer")
    
    expert_swaps = get_number_expert_swaps_harmony()
    df = df.merge(expert_swaps, on="iteration", how="outer")

    save_pd(df, f"../data_processed/systems/router_dynamic_skew/throughput_iters_systems.csv")

num_ranks = 8
num_moe_layers = 12
def get_number_expert_swaps_harmony():
    df = None
    for i in range(num_ranks):
        for j in range(num_moe_layers):
            _df = data.load(f"harmony/{i}/moe_layer-{j}.csv")
            if df is None:
                df = _df[["number expert swaps"]]
            else:
                df = df + _df[["number expert swaps"]]
    df = df.rename(columns={"number expert swaps": "harmony number expert swaps"})
    return df

if __name__ == "__main__":
    create_workload_throughput_iters_system()
