import sys
import pandas as pd
from utils import save_plot
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Pass csv you want to work on")
    exit(0)

path = sys.argv[1]
df = pd.read_csv(path, index_col=0)

df = df[df["layer"] == 4]

num_ranks = df["rank"].max()+1

# Create a pie chart for each rank

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes = axes.flatten()

labels = [
    "Metadata Latency", "First Transfer Latency",
    "Second Transfer Latency", "Computation Latency", "Wait Latency", 
    "Other Latency",
]

legend_wedges = None


for rank in range(num_ranks):
    rank_data = df[df["rank"] == rank].iloc[0]
    values = [
        rank_data["metadata latency (ms)"],
        rank_data["first transfer latency (ms)"],
        rank_data["second transfer latency (ms)"],
        rank_data["comp latency (ms)"],
        rank_data["wait latency (ms)"],
        rank_data["other latency (ms)"],
    ]
    wedges, _, _ = axes[rank].pie(values, autopct='%1.1f%%', startangle=90)
    axes[rank].set_title(f"Rank {rank} Latency Breakdown")

    if legend_wedges is None:
        legend_wedges = wedges

fig.legend(legend_wedges, labels, loc="upper center", ncol=len(labels), fontsize=12)


plt.tight_layout(rect=[0, 0, 1, 0.92])
save_plot(plt, f"../figures/timebreakdown/{path.split('/')[-1].split('.')[0]}.png")
