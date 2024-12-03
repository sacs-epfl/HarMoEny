from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/policies/skew"
data = Data(directory_path)

policies = ["deepspeed", "exflow", "even_split", "harmony", "drop"]
datasets = ["random", "skew25", "skew50", "skew75", "skew90", "skew95"]
num_ranks = 8
num_moe_layers = 12

def create_workload_duration_policy_vs_dataset():
    df = []
    for policy in policies:
        for dataset in datasets:
            _df = data.load(f"{dataset}/{policy}/0/e2e.csv")
            df.append({
                "policy": policy,
                "dataset": dataset,
                "duration (s)": _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/policies/skew/workload_duration_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_policy_vs_dataset()