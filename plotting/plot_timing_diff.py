import pandas as pd
import plotly.express as px
import sys
import json
from datetime import datetime
import os

if len(sys.argv) < 3:
    print("Please provide at least two trace folders you want to plot together")
    exit(1)

moe_layers = [1, 3, 5, 7, 9, 11]

output_folder = f"../plots/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

def plot_e2e(dirs):
    data = []

    for _dir in dirs:
        with open(f"{_dir}/info.json", "r") as f:
            d = json.load(f)
            name = d["scheduler_name"]
        df = pd.read_csv(f"{_dir}/e2e.csv")
        df["name"] = name
        data.append(df)
    
    combined_data = pd.concat(data, ignore_index=True)

    plot = px.line(combined_data, color="name", y="Latency (s)", x="Iteration Number")
    plot.write_image(f"{output_folder}/e2e.png")

def plot_moe_forward_latency(dirs):
    data = []

    for _dir in dirs:
        with open(f"{_dir}/info.json", "r") as f:
            d = json.load(f)
            name = d["scheduler_name"]
        for l in moe_layers:
            df = pd.read_csv(f"{_dir}/moe_l{l}.csv")
            df["name"] = name
            df["layer"] = l
            data.append(df)
    
    combined_data = pd.concat(data, ignore_index=True)

    for l in moe_layers:
        plot = px.line(combined_data[combined_data["layer"] == l], color="name", y="latency (ms)", x="iteration")
        plot.write_image(f"{output_folder}/l{l}.png")

plot_e2e(sys.argv[1:])
plot_moe_forward_latency(sys.argv[1:])

