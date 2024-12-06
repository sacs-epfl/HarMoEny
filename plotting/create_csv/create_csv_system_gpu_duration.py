from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/systems/gpus"
data = Data(directory_path)

systems = ["fastmoe", "fastermoe", "harmony", "exflow"]
gpus = [2, 4, 8]

def create_workload_duration_system_vs_dataset():
    df = []
    for sys in systems:
        for gpu in gpus:
            _df = data.load(f"{gpu}/{sys}/0/e2e.csv")
            df.append({
                "system": sys,
                "num ranks": gpu,
                "duration (s)": _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/systems/gpu/workload_duration_system_vs_gpu.csv")

if __name__ == "__main__":
    create_workload_duration_system_vs_dataset()