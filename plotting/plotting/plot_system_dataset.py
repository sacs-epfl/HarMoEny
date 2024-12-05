import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import numpy as np
from utils import save_plot

systems = ["deepspeed", "exflow", "fastmoe", "fastermoe", "harmony"]

def plot_workload_duration_system_vs_dataset():
    df = pd.read_csv(f"../data_processed/systems/dataset/workload_duration_policy_vs_dataset.csv", index_col=0)

    df["system"] = pd.Categorical(df["system"], categories=systems, ordered=True)

    # Pivot the DataFrame
    pivot_df = df.pivot(index="dataset", columns="system", values="duration (s)")

    # Define the bar width and positions
    bar_width = 0.2
    dataset_gap = 0.5 
    num_policies = len(pivot_df.columns)
    x_positions = []
    current_x = 0

    for _ in range(len(pivot_df.index)):
        x_positions.append(np.arange(current_x, current_x + num_policies * bar_width, bar_width))
        current_x += num_policies * bar_width + dataset_gap

    x_positions = np.array(x_positions)

    fig, ax = plt.subplots(figsize=(12, 5))

    color_palette = colormaps.get_cmap("tab10")  
    system_colors = {system: color_palette(i) for i, system in enumerate(systems)}

    for i, (dataset, positions) in enumerate(zip(pivot_df.index, x_positions)):
        for j, sys in enumerate(pivot_df.columns):
            ax.bar(positions[j], pivot_df.loc[dataset, sys], width=bar_width, color=system_colors[sys], label=sys if i == 0 else "")

    # Flatten x_positions for xticks and add gaps for visualization
    flat_x_positions = [positions.mean() for positions in x_positions]

    ax.set_xticks(flat_x_positions)
    ax.set_xticklabels(pivot_df.index)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Duration")
    ax.legend(title="System")

    save_plot(plt, "../figures/systems/dataset/workload_duration_policy_vs_dataset.png")


if __name__ == "__main__":
    plot_workload_duration_system_vs_dataset()