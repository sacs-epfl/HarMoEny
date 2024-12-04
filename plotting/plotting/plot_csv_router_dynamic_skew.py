import pandas as pd
import matplotlib.pyplot as plt
from utils import save_plot


def plot_workload_throughput_iters_system():
    df = pd.read_csv("../data_processed/systems/router_dynamic_skew/throughput_iters_systems.csv", index_col=0)

    plt.figure(figsize=(12, 6))

    for sys in df.columns:
        plt.plot(df.index.values, df[sys], label=sys)

    plt.legend(title="Systems", loc="lower right")
    plt.xlabel("Iterations")
    plt.ylabel("Throughput (reqs/s)")

    save_plot(plt, "../figures/systems/router_dynamic_skew/throughput_iters_systems.png")

if __name__ == "__main__":
    plot_workload_throughput_iters_system()