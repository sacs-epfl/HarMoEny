import pandas as pd
import plotly.express as px

df = pd.read_csv("exp1_results.csv")
df["percent_diff"] = (df["avg_time_bmm"] / df["avg_time_non_fused"]) * 100
df["time_diff"] = df["avg_time_bmm"] - df["avg_time_non_fused"]


pivot_df = df.pivot(index="num_experts", columns="num_tokens", values="percent_diff")

max_percent = df["percent_diff"].max()
custom_color_scale = [
    [0, "red"],  
    [100/max_percent, "darkgreen"],
    [1, "lightgreen"]
]

heatmap_percent_diff = px.imshow(
    pivot_df,
    labels=dict(x="Number of tokens each expert", y="Number of experts"),
    x=["1", "10", "100", "1,000", "10,000", "100,000"],
    y=["1", "2", "4", "8", "16"],
    color_continuous_scale=custom_color_scale, 
)

heatmap_percent_diff.write_image("exp1_percent_diff.png")

pivot_df_time_diff = df.pivot(index="num_experts", columns="num_tokens", values="time_diff")
custom_color_scale = [
    [0, "red"],  
    [abs(df["time_diff"].min())/df["time_diff"].max(), "yellow"],
    [1, "green"]
]

print(pivot_df_time_diff)

heatmap_time_diff = px.imshow(
    pivot_df_time_diff,
    labels=dict(x="Number of tokens each expert", y="Number of experts"),
    x=["1", "10", "100", "1,000", "10,000", "100,000"],
    y=["1", "2", "4", "8", "16"],
    color_continuous_scale=custom_color_scale, 
    text_auto=True, aspect="auto"
)

heatmap_time_diff.write_image("exp1_time_diff.png")