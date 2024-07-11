import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from theme import update_fig_to_theme
import numpy as np

df = pd.read_csv("outputs/latency_expert_over_num_tokens.csv")

slope, intercept = np.polyfit(df["Num Tokens"], df["Latency (ms)"], 1)
line = slope * df["Num Tokens"] + intercept

fig = px.scatter(df, x="Num Tokens", y="Latency (ms)", title="Latency of an ST8 Expert Forward Pass over num tokens")
fig.add_trace(go.Scatter(x=df["Num Tokens"], y=line, mode="lines", name="Linfit"))
equation_text = f"latency ms = {slope*1000:.3f} ms / (thousand tokens) + {intercept:.2f} ms"
fig.add_annotation(
    x=max(df["Num Tokens"]),
    y=max(line),
    text=equation_text,
    showarrow=False,
    yshift=20
)
update_fig_to_theme(fig)
fig.write_image("plots/expert_latency_over_tokens.png")

# Only first half without linear fit
df = df.iloc[:int(-len(df)/2)]
fig = px.line(df, x="Num Tokens", y="Latency (ms)", title="Latency of an ST8 Expert Forward Pass (FOCUS)")
update_fig_to_theme(fig)
fig.write_image("plots/expert_latency_over_tokens_first_half.png")
