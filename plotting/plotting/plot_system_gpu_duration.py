import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import numpy as np
from utils import save_plot

systems = ["exflow", "fastmoe", "fastermoe", "harmony"]

def plot_workload_duration_system_vs_dataset():
    df = pd.read_csv(f"../data_processed/systems/gpu/workload_duration_system_vs_gpu.csv", index_col=0)

    df["system"] = pd.Categorical(df["system"], categories=systems, ordered=True)

    # Pivot the DataFrame
    pivot_df = df.pivot(index="system", columns="num ranks", values="duration (s)")
    num_ranks = pivot_df.columns

    for index, row in pivot_df.iterrows():
        plt.plot(row.index.values, row.values, label=index)

    plt.xticks(pivot_df.columns)

    plt.xlabel("Number of Ranks")
    plt.ylabel("Duration (s)")
    plt.legend(title="System")

    save_plot(plt, "../figures/systems/gpu/workload_duration_system_vs_gpu.png")


if __name__ == "__main__":
    plot_workload_duration_system_vs_dataset()