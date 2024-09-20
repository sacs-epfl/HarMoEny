import sys
import pandas as pd
import json
import os
import stat
from datetime import datetime

from theme import update_fig_to_theme

import plotly.express as px

OUTPUT_DIR = f"../plots/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"

def create_dir_if_needed():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.chmod(OUTPUT_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} num_gpus paths")

def read_data(_dir: str):
    with open(f"{_dir}/data.json", "r") as f:
        return json.load(f)

def plot_e2e(dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)

        df["name"] = data["name"]

        frames.append(df)

    df = pd.concat(frames)
    df = df.sort_values(["name", "Iteration Number"])
    df = df[df["Iteration Number"] > 3]

    fig = px.line(df, x="Iteration Number", y="Latency (s)", color="name")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e.png")


def plot_imbalance_and_oversubscription(dirs: [str], num_gpus: int):
    final_frames = []
    for _dir in dirs:
        frames = []
        for rank in range(num_gpus):
            df = pd.read_csv(f"{_dir}/{rank}/moe_l1.csv")

            df_select = df[["total number of tokens recv"]]

            df_select = df_select.rename(columns={
                "total number of tokens recv": str(rank)
            })

            if rank == 0:
                df_select = pd.concat((df_select, df[["iteration"]]), axis=1)

            frames.append(df_select)
            
        df_combined = pd.concat(frames, axis=1)

        gpu_columns = list(map(lambda x: str(x), range(num_gpus)))
        df_combined["min"] = df_combined[gpu_columns].min(axis=1)
        df_combined["max"] = df_combined[gpu_columns].max(axis=1)
        df_combined["avg"] = df_combined[gpu_columns].sum(axis=1) // num_gpus
        df_combined["imbalance"] = ((df_combined["max"] - df_combined["min"]) / df_combined["min"]) * 100
        df_combined["oversubscription"] = ((df_combined["max"] - df_combined["avg"]) / df_combined["avg"]) *100


        data = read_data(_dir)
        df_combined["name"] = data["name"]
        final_frames.append(df_combined[["name", "iteration", "imbalance", "oversubscription"]])
    
    df = pd.concat(final_frames)
    df = df.sort_values(["name", "iteration"])
    df = df[df["iteration"] > 3]

    fig = px.line(df, x="iteration", y="imbalance", color="name", labels={"iteration": "Iteration Number", "imbalance": "Imbalance (relative %)"})
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/imbalance.png")


    fig = px.line(df, x="iteration", y="oversubscription", color="name", labels={"iteration": "Iteration Number", "oversubscription": "Oversubscription (relative %)"})
    update_fig_to_theme(fig)

    fig.write_image(f"{OUTPUT_DIR}/oversubscription.png")

def plot_average_speedup(dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)
        df = df.rename(columns={"Latency (s)": data["name"]})
        frames.append(df)
    if len(frames) < 1:
        print("No data to work on, finishing plot_average_speedup")
        return
    comb_df = frames[0]
    for df in frames[1:]:
        comb_df = comb_df.merge(df, on="Iteration Number", how="outer")

    comb_df = comb_df[comb_df["Iteration Number"] > 3]

    columns = comb_df.columns.values[1:]
    if "deepspeed" not in columns:
        print("To obtain speedups you need deepspeed values")
        return

    for col in columns:
        if col == "deepspeed":
            continue
        comb_df[col] = comb_df["deepspeed"] / comb_df[col]
    
    d_avg = []
    for col in columns:
        d_avg.append([comb_df[col].mean()])
    

    avg_df = pd.DataFrame(d_avg, index=columns, columns=["average speedup"])
    avg_df = avg_df.drop(index="deepspeed")
    avg_df = avg_df.sort_index()

    fig = px.bar(avg_df, y="average speedup", labels={"index": "scheduling policy"}, text="average speedup")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/average_speedup.png")

def plot_overall_speedup(dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)
        df = df.rename(columns={"Latency (s)": data["name"]})
        frames.append(df)
    if len(frames) < 1:
        print("No data to work on, finishing plot_overall_speedup")
        return
    comb_df = frames[0]
    for df in frames[1:]:
        comb_df = comb_df.merge(df, on="Iteration Number", how="outer")
    comb_df = comb_df[comb_df["Iteration Number"] > 3]

    columns = comb_df.columns.values[1:]
    if "deepspeed" not in columns:
        print("To obtain speedups you need deepspeed values")
        return
    
    df = comb_df.sum(axis=0)
    df = df.drop(index="Iteration Number")

    for col in columns:
        if col == "deepspeed":
            continue
        df[col] = df["deepspeed"] / df[col]

    df = df.drop(index="deepspeed")
    df = df.sort_index()

    fig = px.bar(df, labels={"index": "scheduling policy", "value": "speedup"}, text="value")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/overall_speedup.png")    


plot_e2e(sys.argv[2:])
plot_imbalance_and_oversubscription(sys.argv[2:], int(sys.argv[1]))
plot_average_speedup(sys.argv[2:])
plot_overall_speedup(sys.argv[2:])