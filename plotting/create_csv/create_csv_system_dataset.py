from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/systems/dataset"
data = Data(directory_path)

systems = ["deepspeed", "fastmoe", "fastermoe", "harmony", "exflow"]
datasets = ["random", "bookcorpus", "wikitext", "wmt19"]

def create_workload_duration_system_vs_dataset():
    df = []
    for sys in systems:
        for dataset in datasets:
            _df = data.load(f"{dataset}/{sys}/0/e2e.csv")
            df.append({
                "system": sys,
                "dataset": dataset,
                "duration (s)": _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/systems/dataset/workload_duration_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_system_vs_dataset()