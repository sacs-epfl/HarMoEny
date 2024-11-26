import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import save_plot

num_moe_layers = 12

metrics = ["avg_gpu_util", "avg_gpu_mem_used", "cpu_util", "cpu_mem_used"]
datasets = ["bookcorpus", "wikitext", "random", "wmt19"]
systems = ["deepspeed", "exflow", "fastmoe", "fastermoe", "harmony"]

def plot_workload_duration_policy_vs_dataset():
    df = pd.read_csv(f"../data_processed/systems/datasets/workload_duration_policy_vs_dataset.csv", index_col=0)

    df["policy"] = pd.Categorical(df["policy"], categories=systems, ordered=True)

    # Pivot the DataFrame
    pivot_df = df.pivot(index="policy", columns="dataset", values="duration (s)")

    # Define the bar width and positions
    bar_width = 0.2
    x = np.arange(len(pivot_df.index))  # Number of systems
    datasets = pivot_df.columns

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, dataset in enumerate(datasets):
        ax.bar(x + i * bar_width, pivot_df[dataset], width=bar_width, label=dataset)

    # Set labels and titles
    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(pivot_df.index)
    ax.set_xlabel("System")
    ax.set_ylabel("Duration")
    ax.legend(title="Dataset")

    save_plot(plt, "../figures/systems/datasets/workload_duration_policy_vs_dataset.png")

def plot_moe_layer_latency_prob_num_tokens_policy_vs_dataset():
    for moe_layer_idx in range(num_moe_layers):
        df = pd.read_csv(f"../data_processed/systems/datasets/layer_latency/layer-{moe_layer_idx}_avg_latency_prob_number_tokens_policy_vs_dataset.csv", index_col=0)

        df["policy"] = pd.Categorical(df["policy"], categories=systems, ordered=True)

        # Pivot the DataFrame
        pivot_df = df.pivot(index="policy", columns="dataset", values="latency (ms)")

        # Define the bar width and positions
        bar_width = 0.2
        x = np.arange(len(pivot_df.index))  # Number of systems
        datasets = pivot_df.columns

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, dataset in enumerate(datasets):
            ax.bar(x + i * bar_width, pivot_df[dataset], width=bar_width, label=dataset)

        # Set labels and titles
        ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(pivot_df.index)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Latency (ms)")
        ax.legend(title="Dataset")

        save_plot(plt, f"../figures/systems/datasets/layer_latency/layer-{moe_layer_idx}_avg_latency_prob_number_tokens_policy_vs_dataset.png")

def plot_avg_imbalance_policy_vs_dataset():
    for moe_layer_idx in range(num_moe_layers):
        df = pd.read_csv(f"../data_processed/systems/datasets/imbalance/layer-{moe_layer_idx}_avg_imbalance_policy_vs_dataset.csv", index_col=0)

        df["policy"] = pd.Categorical(df["policy"], categories=systems, ordered=True)

        # Pivot the DataFrame
        pivot_df = df.pivot(index="policy", columns="dataset", values="avg imbalance")

        # Define the bar width and positions
        bar_width = 0.2
        x = np.arange(len(pivot_df.index))  # Number of systems
        datasets = pivot_df.columns

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, dataset in enumerate(datasets):
            ax.bar(x + i * bar_width, pivot_df[dataset], width=bar_width, label=dataset)

        # Set labels and titles
        ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(pivot_df.index)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Average Imbalance (%)")
        ax.legend(title="Dataset")

        save_plot(plt, f"../figures/systems/datasets/imbalance/layer-{moe_layer_idx}_avg_imbalance_policy_vs_dataset.png")

def plot_avg_metric_policy_vs_dataset():
    for metric in metrics:
        df = pd.read_csv(f"../data_processed/systems/datasets/metrics/avg_metrics_policy_vs_dataset.csv", index_col=0)

        df["policy"] = pd.Categorical(df["policy"], categories=systems, ordered=True)

        # Pivot the DataFrame
        pivot_df = df.pivot(index="policy", columns="dataset", values=metric)

        # Define the bar width and positions
        bar_width = 0.2
        x = np.arange(len(pivot_df.index))  # Number of systems
        datasets = pivot_df.columns

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, dataset in enumerate(datasets):
            ax.bar(x + i * bar_width, pivot_df[dataset], width=bar_width, label=dataset)

        # Set labels and titles
        ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(pivot_df.index)
        ax.set_xlabel("Policy")
        ax.set_ylabel(metric)
        ax.legend(title="Dataset")

        save_plot(plt, f"../figures/systems/datasets/metrics/avg_{metric}_policy_vs_dataset.png")

def plot_metric_over_time_policy():
    for dataset in datasets:
        for metric in metrics:
            df = pd.read_csv(f"../data_processed/systems/datasets/metrics/{dataset}_policy_metrics_over_iters.csv", index_col=0)

            plt.figure(figsize=(12, 6))

            for policy, group in df.groupby("policy"):
                plt.plot(group["seconds"], group["avg_gpu_util"], label=policy)
            
            plt.xlabel("Seconds")
            plt.ylabel(metric)
            plt.legend(title="Policy")
            plt.grid(True)

            save_plot(plt, f"../figures/systems/datasets/metrics/{dataset}_policy_{metric}_over_iters.png")

if __name__ == "__main__":
    plot_workload_duration_policy_vs_dataset()
    # plot_moe_layer_latency_prob_num_tokens_policy_vs_dataset()
    # plot_avg_imbalance_policy_vs_dataset()
    # plot_avg_metric_policy_vs_dataset()
    # plot_metric_over_time_policy()