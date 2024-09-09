import sys
import pandas as pd
import json
import os
import stat
from datetime import datetime

import plotly.express as px

OUTPUT_DIR = f"../plots/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chmod(OUTPUT_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} num_gpus paths")

def read_data(_dir: str):
    with open(f"{_dir}/data.json", "r") as f:
        return json.load(f)

def plot_e2e(dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)

        df["name"] = data["name"]

        frames.append(df)

    df = pd.concat(frames)

    fig = px.line(df, x="Iteration Number", y="Latency (s)", color="name")
    fig.write_image(f"{OUTPUT_DIR}/e2e.png")


def plot_imbalance(dirs: [str], num_gpus: int):
    final_frames = []
    for _dir in dirs:
        frames = []
        for rank in range(num_gpus):
            df = pd.read_csv(f"{_dir}/{rank}/moe_l1.csv")

            if rank == 0:
                df = df[["iteration", f"gpu:{rank} comp num tokens"]]
            else:
                df = df[[f"gpu:{rank} comp num tokens"]]

            frames.append(df)
            
        df_combined = pd.concat(frames, axis=1)
        gpu_columns = [f"gpu:{i} comp num tokens" for i in range(num_gpus)]
        df_combined["min_comp"] = df_combined[gpu_columns].min(axis=1)
        df_combined["max_comp"] = df_combined[gpu_columns].max(axis=1)
        df_combined["avg_comp"] = df_combined[gpu_columns].sum(axis=1) // num_gpus
        df_combined["imbalance"] = ((df_combined["max_comp"] - df_combined["avg_comp"]) / df_combined["avg_comp"]) *100
        data = read_data(_dir)
        df_combined["name"] = data["name"]
        final_frames.append(df_combined[["name", "iteration", "imbalance"]])
    
    df = pd.concat(final_frames)

    fig = px.line(df, x="iteration", y="imbalance", color="name")
    fig.write_image(f"{OUTPUT_DIR}/imbalance.png")


plot_e2e(sys.argv[2:])
plot_imbalance(sys.argv[2:], int(sys.argv[1]))

# for _dir in sys.argv[2:]:
#     df = pd.read_csv(f"{_dir}/0/e2e.csv")
#     data = read_data(_dir)

#     df["name"] = data["name"]

#     print(df)

