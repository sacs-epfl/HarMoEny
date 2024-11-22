import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import save_plot

num_moe_layers = 12

def plot_layer_latency_moe_vs_dense_vs_dense_param_matched():
    for moe_layer_idx in range(num_moe_layers):
        df = pd.read_csv(f"../data_processed/experiment_1/latency_layer-{moe_layer_idx}.csv")

        plt.figure(figsize=(12, 6))

        for _type, group in df.groupby("type"):
            plt.plot(group["iteration"], group["latency (ms)"], label=_type)
        
        plt.xlabel("Iteration")
        plt.ylabel(f"Layer {moe_layer_idx} latency")
        plt.legend()
        plt.grid(True)

        save_plot(plt, f"../figures/experiment_1/layer_latency/latency_layer-{moe_layer_idx}.png")

def plot_expert_token_distribution():
    for moe_layer_idx in range(num_moe_layers):
        df = pd.read_csv(f"../data_processed/experiment_1/expert_token_distribution_layer-{moe_layer_idx}.csv")

        plt.figure(figsize=(12, 6))

        row = df.iloc[len(df) // 2]
        plt.bar(range(len(row)), row)

        plt.xlabel("Expert Index")
        plt.ylabel("Token proportion (%)")
        
        save_plot(plt, f"../figures/experiment_1/expert_token_distribution/expert_token_distribution_layer-{moe_layer_idx}.png")

def plot_working_to_idle_time():
    for moe_layer_idx in range(num_moe_layers):
        df = pd.read_csv(f"../data_processed/experiment_1/working_to_idle_time_layer-{moe_layer_idx}.csv")

        plt.figure(figsize=(12, 6))

        plt.plot(df["iteration"], df["min (ms)"], label="Quickest GPU comp time")
        plt.plot(df["iteration"], df["idle (ms)"], label="Quickest GPU idle time")

        plt.xlabel("Iteration")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)

        save_plot(plt, f"../figures/experiment_1/working_to_idle_time/working_to_idle_time_layer-{moe_layer_idx}.png")


if __name__ == "__main__":
    #plot_layer_latency_moe_vs_dense_vs_dense_param_matched()
    #plot_expert_token_distribution()
    plot_working_to_idle_time()