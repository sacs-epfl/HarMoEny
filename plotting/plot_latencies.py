import plotly.express as px
import pandas as pd

df = pd.read_csv("outputs/latency_measurements_naive_expert_parallel_switch_8_ORIG.csv")
df = df[df["Iteration Number"] != 0] 

fig = px.line(df, x="Iteration Number", y="Latency (s)", color="Number of Tokens")
fig.write_image("plots/naive_latencies_over_iters.png")

df_1 = df[df["Number of Tokens"] == 245760]
fig = px.line(df_1, x="Iteration Number", y="Latency (s)")
fig.write_image("plots/naive_latencies_over_iters_last.png")

# Tail Latency
df_2 = df
# Calculate percentiles (P50, P90, P95) for each Number of Tokens
percentiles = [50, 90, 95]
percentile_df = df.groupby("Number of Tokens")["Latency (s)"].quantile(q=[p/100 for p in percentiles], interpolation="nearest").unstack().reset_index()
percentile_df.columns = ["Number of Tokens", "P50", "P90", "P95"]

mean_df = df.groupby("Number of Tokens")["Latency (s)"].mean().reset_index()
mean_df.columns = ["Number of Tokens", "Mean"]

result_df = pd.merge(percentile_df, mean_df, on="Number of Tokens")

fig = px.line(result_df, x="Number of Tokens", y=["P50", "P90", "P95", "Mean"],
              labels={"value": "Latency (s)", "Number of Tokens": "Number of Tokens"})
fig.write_image("plots/naive_percentile_latencies.png")

fig = px.line(result_df[:int(-result_df.shape[0]/2)], x="Number of Tokens", y=["P50", "P90", "P95", "Mean"],
              labels={"value": "Latency (s)", "Number of Tokens": "Number of Tokens"})
fig.write_image("plots/naive_percentile_latencies_first_half.png")

fig = px.line(result_df[int(-result_df.shape[0]/5):], x="Number of Tokens", y=["P50", "P90", "P95", "Mean"],
              labels={"value": "Latency (s)", "Number of Tokens": "Number of Tokens"})
fig.write_image("plots/naive_percentile_latencies_last_fifth.png")