import pandas as pd
import matplotlib.pyplot as plt
from utils import save_plot


def plot_workload_throughput_iters_system():
    df = pd.read_csv("../data_processed/systems/router_dynamic_skew/throughput_iters_systems.csv", index_col=0)

    plt.figure(figsize=(12, 6))

    # Create the primary axis
    ax1 = plt.gca()
    for sys in df.columns:
        if sys != "harmony number expert swaps":
            ax1.plot(df.index.values, df[sys], label=sys)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Throughput (reqs/s)")
    ax1.legend(title="Systems", loc="lower left")

    # Create the secondary axis
    ax2 = ax1.twinx()
    if "harmony number expert swaps" in df.columns:
        ax2.plot(df.index.values, df["harmony number expert swaps"], color='orange', linestyle='--', label="Harmony")
        ax2.set_ylabel("Number Expert Swaps", color='orange')
        ax2.tick_params(axis='y', colors='orange')
        ax2.set_ylim(0, 200)
        ax2.legend(title="System", loc="lower right")

    # for sys in df.columns:
    #     plt.plot(df.index.values, df[sys], label=sys)

    # plt.legend(title="Systems", loc="lower right")
    # plt.xlabel("Iterations")
    # plt.ylabel("Throughput (reqs/s)")

    save_plot(plt, "../figures/systems/router_dynamic_skew/throughput_iters_systems.png")

if __name__ == "__main__":
    plot_workload_throughput_iters_system()