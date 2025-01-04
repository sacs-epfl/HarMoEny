from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../outputs/exp-policies-dataset"
data = Data(directory_path)

policies = ["deepspeed", "exflow", "even_split", "harmony", "drop"]
datasets = ["random", "bookcorpus", "wmt19", "wikitext"]
num_ranks = 8

def create_workload_duration_policy_vs_dataset():
    df = []
    for policy in policies:
        for dataset in datasets:
            _df = data.load(f"{dataset}/{policy}/0/e2e.csv")
            meta = data.read_meta(f"{dataset}/{policy}")
            df.append({
                "policy": policy,
                "dataset": dataset,
                "duration (s)": _df["latency (s)"].sum(axis=0),
                "throughput (toks/s)": meta["num_samples"] / _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/policies/dataset/workload_duration_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_policy_vs_dataset()