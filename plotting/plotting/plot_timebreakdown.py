import sys
import pandas as pd
from utils import save_plot
import matplotlib.pyplot as plt

# COLOUR_MAP = {
#     "metadata latency (ms)": "#1f77b4", 
#     "schedule latency (ms)": "#2ca02c",  
#     "first transfer latency (ms)": "#9467bd", 
#     "second transfer latency (ms)": "#ff7f0e",  
#     "comp latency (ms)": "#d62728",  
#     "wait latency (ms)": "#17becf",  
#     "other latency (ms)": "#8b4513", 
# }

COLOUR_MAP = {
    "all_to_all communication latency (ms)": "#1f77b4", 
    "schedule latency (ms)": "#2ca02c",  
    "comp latency (ms)": "#d62728",  
    "wait latency (ms)": "#17becf",  
    "other latency (ms)": "#8b4513", 
}

DEFAULT_COLOUR = "#7f7f7f"  

if len(sys.argv) < 2:
    print("Pass csv you want to work on")
    exit(0)

path = sys.argv[1]
df = pd.read_csv(path, index_col=0)

df = df[df["layer"] == 4]


############# COMBINING THE THREE COMMS ###########
df["all_to_all communication latency (ms)"] = (
    df["first transfer latency (ms)"] + 
    df["second transfer latency (ms)"] + 
    df["metadata latency (ms)"]
)

df = df.drop(columns=["first transfer latency (ms)", "second transfer latency (ms)", "metadata latency (ms)"])

columns = ["all_to_all communication latency (ms)"] + [col for col in df.columns if col != "all_to_all communication latency (ms)"]
df = df[columns]
####################################################

num_ranks = df["rank"].max()+1

# Create a pie chart for each rank
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes = axes.flatten()

labels = df.columns.to_list()
labels.remove("layer")
labels.remove("latency (ms)")
labels.remove("rank")

legend_wedges = None

for rank in range(num_ranks):
    rank_data = df[df["rank"] == rank].iloc[0]
    total_latency = rank_data["latency (ms)"]
    rank_data = rank_data.drop(["layer", "rank", "latency (ms)"])
    values = rank_data.tolist()
    component_labels = rank_data.index.tolist()

    colors = [COLOUR_MAP.get(component, DEFAULT_COLOUR) for component in component_labels]

    wedges, _, _ = axes[rank].pie(values, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[rank].set_title(f"Rank {rank} Latency Breakdown\nTotal Latency: {total_latency:.2f} ms")

    if legend_wedges is None:
        legend_wedges = wedges

fig.legend(legend_wedges, labels, loc="upper center", ncol=len(labels), fontsize=12)


plt.tight_layout(rect=[0, 0, 1, 0.92])
save_plot(plt, f"../figures/timebreakdown/{path.split('/')[-1].split('.')[0]}.png")