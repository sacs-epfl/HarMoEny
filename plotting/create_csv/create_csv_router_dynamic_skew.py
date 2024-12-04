from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/systems/router_dynamic_skew"
data = Data(directory_path)

systems = ["deepspeed", "fastmoe", "fastermoe", "harmony", "exflow"]

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

    save_pd(df, f"../data_processed/systems/router_dynamic_skew/throughput_iters_systems.csv")

if __name__ == "__main__":
    create_workload_throughput_iters_system()