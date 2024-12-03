from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/systems/router_skew"
data = Data(directory_path)

systems = ["deepspeed", "fastmoe", "fastermoe", "harmony", "exflow"]
skews = ["0.0", "0.5", "0.95"]

predictions = {
    "deepspeed": {
        "0.5": 4807.8,
        "0.95": 8987.7,
    }
}

def create_workload_duration_policy_vs_dataset():
    df = []
    for sys in systems:
        for skew in skews:
            # Check if exists 
            path = f"{skew}/{sys}/0/e2e.csv"
            if data.exists(path):
                _df = data.load(path)
                df.append({
                    "system": sys,
                    "skew": skew,
                    "duration (s)": _df["latency (s)"].sum(axis=0)
                })
            elif sys in predictions and skew in predictions[sys]:
                df.append({
                    "system": sys, 
                    "skew": skew,
                    "duration (s)": predictions[sys][skew]
                })
            else:
                print(f"No value for system {sys} at skew {skew}")
                

    save_pd(pd.DataFrame(df), "../data_processed/systems/router_skew/workload_duration_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_policy_vs_dataset()