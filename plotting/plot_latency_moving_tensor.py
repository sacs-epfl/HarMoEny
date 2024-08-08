import plotly.express as px
import pandas as pd
from theme import update_fig_to_theme
import ast
import numpy as np

df = pd.read_csv("outputs/latencies_moving_expert.csv")

# print(df["latency (ms)"])
# print(df["latency (ms)"].apply(ast.literal_eval).apply(np.mean))
df["latency (ms)"] = df["latency (ms)"].apply(ast.literal_eval).apply(np.mean)
heatmap_data = df.pivot(index="x", columns="y", values="latency (ms)")
heatmap_data = heatmap_data.iloc[::-1]
fig = px.imshow(heatmap_data, labels=dict(x="Source", y="Destination", color="Latency (ms)"), color_continuous_scale='viridis')
update_fig_to_theme(fig)
fig.write_image("plots/latencies_moving_expert.png")

