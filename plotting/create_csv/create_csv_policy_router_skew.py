from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/policies/router_skew"
data = Data(directory_path)

policies = ["deepspeed", "exflow", "even_split", "harmony", "drop"]
skews = [0.0, 0.5, 0.95]
num_ranks = 8

def create_workload_duration_policy_vs_dataset():
    df = []
    for policy in policies:
        for skew in skews:
            _df = data.load(f"{skew}/{policy}/0/e2e.csv")
            df.append({
                "policy": policy,
                "skew": skew,
                "duration (s)": _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/policies/router_skew/workload_duration_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_policy_vs_dataset()