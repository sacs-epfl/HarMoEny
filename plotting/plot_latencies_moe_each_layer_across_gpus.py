import plotly.express as px
import pandas as pd
from theme import update_fig_to_theme
import os

df_1 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l1.csv")
df_3 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l3.csv")
df_5 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l5.csv")
df_7 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l7.csv")
df_9 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l9.csv")
df_11 = pd.read_csv(f"outputs/latency_measurements_naive_expert_parallel_switch_8/moe_l11.csv")

df_1_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l1.csv")
df_3_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l3.csv")
df_5_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l5.csv")
df_7_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l7.csv")
df_9_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l9.csv")
df_11_demeter = pd.read_csv(f"outputs/latency_measurements_demeter_expert_parallel_switch_8/moe_l11.csv")

def print_info_regarding(df, layer_num):
    DIR = f"plots/switch-transformer-8-demeter/{layer_num}"
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    df = df[df["iteration"] != 0]
    _df_time = df[["gpu:0 latency (ms)", "gpu:1 latency (ms)", "gpu:2 latency (ms)", "gpu:3 latency (ms)"]]
    _df = df[["total number of tokens", "iteration"]]
    _df["min"] = _df_time.min(axis=1)
    _df["max"] = _df_time.max(axis=1)
    _df.loc[:, "diff"] = _df.loc[:, "max"] - _df.loc[:, "min"]
    _df.loc[:, "multiples"] = _df.loc[:, "diff"] / _df.loc[:, "min"]

    # Everything
    ## Everything diff between first and last GPU
    fig = px.line(_df, x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "last GPU to finish - first (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.update_yaxes(range=[0,160])
    fig.write_image(f"{DIR}/gpu_latency_delta_all_l{layer_num}.png")

    ## Everything first GPU
    fig = px.line(_df, x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "time first GPU finish (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_latency_min_all_l{layer_num}.png")

    ## Everything ratio length GPU
    fig = px.line(_df, x="iteration", y="multiples", color="total number of tokens", 
            labels={"iteration": "iteration number", "multiples": "relative difference fastest GPU to slowest", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.update_yaxes(range=[0,10])
    fig.write_image(f"{DIR}/gpu_latency_ratio_all_l{layer_num}.png")

    # Focus
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "last GPU to finish - first (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.update_yaxes(range=[0,4.5])
    fig.write_image(f"{DIR}/gpu_latency_delta_focus_l{layer_num}.png")

    ## Focus first GPU
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "time first GPU finish (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_latency_min_focus_l{layer_num}.png")

    ## Focus ratio length GPU
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="multiples", color="total number of tokens", 
            labels={"iteration": "iteration number", "multiples": "relative difference fastest GPU to slowest", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.update_yaxes(range=[0,3])
    fig.write_image(f"{DIR}/gpu_latency_ratio_focus_l{layer_num}.png")


    # Focuser
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "last GPU to finish - first (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.update_yaxes(range=[0,1.6])
    fig.write_image(f"{DIR}/gpu_latency_delta_focuser_l{layer_num}.png")

    ## Focuser first GPU
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "time first GPU finish (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_latency_min_focuser_l{layer_num}.png")

    ## Focuser ratio length GPU
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="multiples", color="total number of tokens", 
            labels={"iteration": "iteration number", "multiples": "relative difference fastest GPU to slowest", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.update_yaxes(range=[0,1.5])
    fig.write_image(f"{DIR}/gpu_latency_ratio_focuser_l{layer_num}.png")

    # Mean Variance Work
    grouped_stats = _df.groupby('total number of tokens').agg({'diff': ['mean', 'var'], 'multiples': ['mean', 'var']}).reset_index()
    grouped_stats.columns = [' '.join(col).strip() for col in grouped_stats.columns.values]

    ## Diff
    fig = px.line(grouped_stats, x="total number of tokens", y="diff mean",
            labels={"diff mean": "average difference between first and last GPU (ms)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.update_yaxes(range=[0,100])
    fig.write_image(f"{DIR}/gpu_latency_delta_mean_all_l{layer_num}.png")

    fig = px.line(grouped_stats, x="total number of tokens", y="diff var",
            labels={"diff var": "variance difference between first and last GPU (ms^2)", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.update_yaxes(range=[0,300])
    fig.write_image(f"{DIR}/gpu_latency_delta_variance_all_l{layer_num}.png")

    ## Ratio
    fig = px.line(grouped_stats, x="total number of tokens", y="multiples mean",
            labels={"multiples mean": "average relative difference fastest GPU to slowest", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.write_image(f"{DIR}/gpu_latency_ratio_mean_all_l{layer_num}.png")

    fig = px.line(grouped_stats, x="total number of tokens", y="multiples var",
            labels={"multiples var": "variance relative difference fastest GPU to slowest", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=20,
    )
    fig.write_image(f"{DIR}/gpu_latency_ratio_variance_all_l{layer_num}.png")






    # Token Imbalance
    _df_tokens = df[["gpu:0 num tokens", "gpu:1 num tokens", "gpu:2 num tokens", "gpu:3 num tokens"]]
    _df = df[["total number of tokens", "iteration"]]
    _df["min"] = _df_tokens.min(axis=1)
    _df["max"] = _df_tokens.max(axis=1)
    _df.loc[:, "diff"] = _df.loc[:, "max"] - _df.loc[:, "min"]
    _df.loc[:, "multiples"] = _df.loc[:, "diff"] / _df.loc[:, "min"]

    ## Everything
    fig = px.line(_df, x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "most tokens received by 1 GPU minus least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_tokens_delta_all_l{layer_num}.png")

    ### Min
    fig = px.line(_df, x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "minimum number of tokens received by a GPU", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=22,
    )
    fig.write_image(f"{DIR}/gpu_tokens_min_all_l{layer_num}.png")

    ### Multiples
    fig = px.line(_df, x="iteration", y="multiples", color="total number of tokens",
            labels={"iteration": "iteration number", "multiples": "relative difference most popular GPU to least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=18,
    )
    fig.write_image(f"{DIR}/gpu_tokens_ratio_all_l{layer_num}.png")

    ## Focus
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "most tokens received by 1 GPU minus least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_tokens_delta_focus_l{layer_num}.png")

    ### Min
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "minimum number of tokens received by a GPU", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=22,
    )
    fig.write_image(f"{DIR}/gpu_tokens_min_focus_l{layer_num}.png")

    ### Multiples
    fig = px.line(_df[_df["total number of tokens"] <= 7680], x="iteration", y="multiples", color="total number of tokens",
            labels={"iteration": "iteration number", "multiples": "relative difference most popular GPU to least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUS)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=18,
    )
    fig.write_image(f"{DIR}/gpu_tokens_ratio_focus_l{layer_num}.png")

    ## Focuser
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="diff", color="total number of tokens",
            labels={"iteration": "iteration number", "diff": "most tokens received by 1 GPU minus least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        }
    )
    fig.write_image(f"{DIR}/gpu_tokens_delta_focuser_l{layer_num}.png")

    ### Min
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="min", color="total number of tokens",
            labels={"iteration": "iteration number", "min": "minimum number of tokens received by a GPU", "total number of tokens": "num tokens"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=22,
    )
    fig.write_image(f"{DIR}/gpu_tokens_min_focuser_l{layer_num}.png")

    ### Multiples
    fig = px.line(_df[_df["total number of tokens"] <= 960], x="iteration", y="multiples", color="total number of tokens",
            labels={"iteration": "iteration number", "multiples": "relative difference most popular GPU to least", "total number of tokens": "num tokens (prompt)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE) (FOCUSER)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=18,
    )
    fig.write_image(f"{DIR}/gpu_tokens_ratio_focuser_l{layer_num}.png")


def print_analysis(df, layer_num):
    DIR = f"plots/switch-transformer-8-demeter/{layer_num}"
    if not os.path.exists(DIR):
        os.makedirs(DIR)


    df = df[df["iteration"] != 0]
    _df_time = df[["gpu:0 latency (ms)", "gpu:1 latency (ms)", "gpu:2 latency (ms)", "gpu:3 latency (ms)"]]
    _df = df[["iteration"]]
    _df["min"] = _df_time.min(axis=1)
    _df["max"] = _df_time.max(axis=1)
    df["diff latency"] = _df["max"] - _df["min"]

    _df_tokens = df[["expert_0 num tokens", "expert_1 num tokens", "expert_2 num tokens", "expert_3 num tokens", 
        "expert_4 num tokens", "expert_5 num tokens", "expert_6 num tokens", "expert_7 num tokens"]]
    _df = df[["iteration"]]
    _df["min"] = _df_tokens.min(axis=1)
    _df["max"] = _df_tokens.max(axis=1)
    df["diff tokens"] = _df["max"] - _df["min"]


    fig = px.scatter(df, x="diff tokens", y="diff latency",
            labels={"diff tokens": "token diff between GPU that received most and least", "diff latency": "time diff between GPU that took the longest and the shortest (ms)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
        yaxis_title_font_size=18,
    )
    fig.update_yaxes(range=[0,100])
    fig.write_image(f"{DIR}/correlation_l{layer_num}.png")

def print_tails(df, layer_num):
    DIR = f"plots/switch-transformer-8-demeter/{layer_num}"
    if not os.path.exists(DIR):
        os.makedirs(DIR)


    df = df[df["iteration"] != 0]
    _df_time = df[["gpu:0 latency (ms)", "gpu:1 latency (ms)", "gpu:2 latency (ms)", "gpu:3 latency (ms)"]]

    # Let us get the slowest for each iteration since that is what we rely on
    df["max"] = _df_time.max(axis=1)
    mean_df = df.groupby("total number of tokens")["max"].mean().reset_index()
    mean_df.columns = ["total number of tokens", "mean"]

    percentiles = [50, 90, 95, 99]
    percentiles_df = df.groupby("total number of tokens")["max"].quantile(q=[p/100 for p in percentiles],
        interpolation="nearest").unstack().reset_index()
    percentiles_df.columns = ["total number of tokens", "P50", "P90", "P95", "P99"]

    result_df = pd.merge(mean_df, percentiles_df, on="total number of tokens")


    fig = px.line(result_df, x="total number of tokens", y=["mean", "P50", "P90", "P95", "P99"],
            labels={"total number of tokens": "Number of tokens in prompt", "value": "Latency (ms)"}, 
            title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)"
        )
    update_fig_to_theme(fig)
    fig.update_layout(
        title={
            "font": {
                "size": 25
            }
        },
        legend={
            "font": {
                "size": 20,
            }
        },
    )
    fig.update_yaxes(range=[0,160])
    fig.write_image(f"{DIR}/latency_distribution_l{layer_num}.png")


def print_comparison(naive, demeter, layer_num):
    DIR = f"plots/switch-transformer-8-comparison/{layer_num}"
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    _naive_tokens = naive[["gpu:0 num tokens", "gpu:1 num tokens", "gpu:2 num tokens", "gpu:3 num tokens"]]
    _naive = naive[["total number of tokens", "iteration"]]
    _naive["min"] = _naive_tokens.min(axis=1)
    _naive["max"] = _naive_tokens.max(axis=1)
    _naive["diff"] = _naive["max"] - _naive["min"]

    _demeter_tokens = demeter[["gpu:0 num tokens", "gpu:1 num tokens", "gpu:2 num tokens", "gpu:3 num tokens"]]
    _demeter = demeter[["total number of tokens", "iteration"]]
    _demeter["min"] = _demeter_tokens.min(axis=1)
    _demeter["max"] = _demeter_tokens.max(axis=1)
    _demeter["diff"] = _demeter["max"] - _demeter["min"]


    df = naive[["total number of tokens", "iteration"]]
    df["percent reduction"] = ((_naive["diff"] - _demeter["diff"]) / _naive["diff"]) * 100

    fig = px.line(df, x="iteration", y="percent reduction", color="total number of tokens",
        labels={"iteration": "iteration number", "percent reduction": "% GPU offbalance reduction", "total number of tokens": "num tokens"}, 
        title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)")
    update_fig_to_theme(fig)
    fig.write_image(f"{DIR}/token_reduction_{layer_num}.png")

    df_mean = df.groupby(by="total number of tokens")["percent reduction"].mean().reset_index()
    fig = px.line(df_mean, x="total number of tokens", y="percent reduction",
        labels={"percent reduction": "% GPU offbalance reduction", "total number of tokens": "num tokens"},
        title=f"4 GPU naive expert parallelism SwitchTransformer Layer {layer_num} (MoE)",
        markers=True)
    update_fig_to_theme(fig)
    fig.write_image(f"{DIR}/token_reduction_average_{layer_num}.png")
    


# print_info_regarding(df_1, 1)
# print_info_regarding(df_3, 3)
# print_info_regarding(df_5, 5)
# print_info_regarding(df_7, 7)
# print_info_regarding(df_9, 9)
# print_info_regarding(df_11, 11)

# print_analysis(df_1, 1)
# print_analysis(df_3, 3)
# print_analysis(df_5, 5)
# print_analysis(df_7, 7)
# print_analysis(df_9, 9)
# print_analysis(df_11, 11)

# print_tails(df_1, 1)
# print_tails(df_3, 3)
# print_tails(df_5, 5)
# print_tails(df_7, 7)
# print_tails(df_9, 9)
# print_tails(df_11, 11)

print_comparison(df_1, df_1_demeter, 1)

