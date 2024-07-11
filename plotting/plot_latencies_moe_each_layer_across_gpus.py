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



def print_info_regarding(df, layer_num):
    DIR = f"plots/switch-transformer-8/{layer_num}"
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
    print(df)
    _df_tokens = df[["expert_0 num tokens", "expert_1 num tokens", "expert_2 num tokens", "expert_3 num tokens", 
        "expert_4 num tokens", "expert_5 num tokens", "expert_6 num tokens", "expert_7 num tokens"]]
    _df = df[["total number of tokens", "iteration"]]
    _df_tokens["gpu:0 num tokens"] = _df_tokens["expert_0 num tokens"] + _df_tokens["expert_1 num tokens"]
    _df_tokens["gpu:1 num tokens"] = _df_tokens["expert_2 num tokens"] + _df_tokens["expert_3 num tokens"]
    _df_tokens["gpu:2 num tokens"] = _df_tokens["expert_4 num tokens"] + _df_tokens["expert_5 num tokens"]
    _df_tokens["gpu:3 num tokens"] = _df_tokens["expert_6 num tokens"] + _df_tokens["expert_7 num tokens"]    
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


print_info_regarding(df_1, 1)
print_info_regarding(df_3, 3)
print_info_regarding(df_5, 5)
print_info_regarding(df_7, 7)
print_info_regarding(df_9, 9)
print_info_regarding(df_11, 11)


