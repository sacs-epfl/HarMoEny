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


def plot_imbalance_and_oversubscription(dirs: [str]):
    final_frames = []
    for _dir in dirs:
        data = read_data(_dir)
        num_gpus = int(data["world_size"])

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

    df = df.sort_index()

    df_speedup = df["deepspeed"] / df

    plot_df = pd.DataFrame({
        'scheduling_policy': df.index,
        'workload_duration': df.values,
        'speedup': df_speedup.values,
        'label': [f"{value:.2f} ({speedup:.2f}x)" for value, speedup in zip(df, df_speedup)]
    })
    plot_df['label'] = plot_df['label'].astype(str)

    fig = px.bar(plot_df, 
                 x='scheduling_policy', 
                 y='workload_duration', 
                 text='label',
                 labels={"scheduling_policy": "scheduling policy", 
                         "workload_duration": "workload duration (s)"})
    
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/overall_e2e.png")    

def plot_throughput(dirs: [str]):
    throughput = {"policy": [], "throughput": []}
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        df = df[df["Iteration Number"] > 3]
        num_iters = len(df)
        df = df.sum(axis=0)
        data = read_data(_dir)
        num_samples = int(data["world_size"]) * int(data["batch_size"]) * num_iters

        throughput["policy"].append(data["name"])
        throughput["throughput"].append(num_samples / df["Latency (s)"])
    
    df = pd.DataFrame(throughput)

    if len(df[df["policy"]=="deepspeed"]) == 0:
        print("To obtain throughput you need deepspeed values")
        return

    deepspeed = df[df["policy"]=="deepspeed"]["throughput"].iloc[0]
    df["labels"] = [f"{val:.2f} ({val/deepspeed:.2f}x)" for val in df["throughput"].tolist()]

    fig = px.bar(df, x="policy", y="throughput", text="labels", 
        labels={"policy": "scheduling policy", "throughput": "throughput (reqs/s)"})
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/throughput.png")

def plot_maximal_batch_size(path: str):
    df = pd.read_csv(path)

    for time in df["time"].tolist():
        df_i = df[df["time"]==time]
        df_i = df_i.drop(columns=["time"])
        plot_df_i = df_i.melt(var_name='policy_name', value_name='batch_size')
        plot_df_i = plot_df_i.sort_values(by="policy_name")
        fig = px.bar(plot_df_i,
                title=f"Maximal batch size with {time}s requirement",
                x="policy_name",
                y="batch_size",
                text="batch_size",
                labels={"policy_name": "scheduling policy",
                        "batch_size": "maximal batch size"})
        fig.update_traces(textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
        update_fig_to_theme(fig)

        create_dir_if_needed()
        fig.write_image(f"{OUTPUT_DIR}/maximal_batch.png")


def plot_speedup_across_metric(metric: str, dirs: [str]):
    values = []

    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        df = df[df["Iteration Number"] > 3]
        df = df.sum(axis=0)
        data = read_data(_dir)
        values.append({
            "metric": data[metric],
            "policy": data["name"],
            "time": df["Latency (s)"]
        })

    df = pd.DataFrame(values)
    metrics = df["metric"].unique()
    df['speedup'] = float('nan')
    for m in metrics:
        deepspeed_value = df[(df["metric"] == m) & (df["policy"] == "deepspeed")]["time"].iloc[0]
        speedup_values = deepspeed_value / df[df["metric"] == m]["time"]
        df.loc[df["metric"] == m, "speedup"] = speedup_values
    
    df = df.sort_values("policy", axis=0)
    df["labels"] = [f"{val:.2f}" for val in df["speedup"].tolist()]
    fig = px.scatter(df, x="policy", y="speedup", color="metric", text="labels", labels={"metric": metric})
    fig.update_traces(textposition='middle right', marker_size=20)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/speedup_across_{metric}.png")

def plot_imbalance_and_oversubscription_across_metric(metric: str, dirs: [str]):
    values = []

    for _dir in dirs:
        data = read_data(_dir)

        interm2 = []

        for z in [1,3,5,7,9,11]: 
            interm = []

            ranks = list(range(int(data["world_size"])))

            for i in ranks:
                df = pd.read_csv(f"{_dir}/{i}/moe_l{z}.csv")
                df = df[df["iteration"] > 3]

                df = df[["total number of tokens recv"]]
                df = df.rename(columns={"total number of tokens recv": i})

                interm.append(df)
            
            df = interm[0]
            for _d in interm[1:]:
                df = df.join(_d)
            
            df["max"] = df[ranks].max(axis=1)
            df["min"] = df[ranks].min(axis=1)
            df["avg"] = df[ranks].mean(axis=1)

            df["imbalance"] = ((df["max"] - df["min"]) / df["min"]) * 100
            df["oversubscription"] = ((df["max"] - df["avg"]) / df["avg"]) * 100


            interm2.append({
                "layer": z,
                "imbalance": df["imbalance"].mean(axis=0),
                "oversubscription": df["oversubscription"].mean(axis=0),
            })
        
        df = pd.DataFrame(interm2)

        values.append({
            "metric": data[metric],
            "policy": data["name"],
            "imbalance": df["imbalance"].mean(axis=0),
            "oversubscription": df["oversubscription"].mean(axis=0)
        })

    df = pd.DataFrame(values)

    df = df.sort_values("policy", axis=0)

    df["labels_imbalance"] = [f"{val:.2f}" for val in df["imbalance"].tolist()]
    df["labels_oversubscription"] = [f"{val:.2f}" for val in df["oversubscription"].tolist()]
    
    fig = px.scatter(df, x="policy", y="imbalance", color="metric", text="labels_imbalance", labels={"metric": metric})
    fig.update_traces(textposition='middle right', marker_size=20)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/imbalance_across_{metric}.png")

    fig = px.scatter(df, x="policy", y="oversubscription", color="metric", text="labels_oversubscription")
    fig.update_traces(textposition='middle right', marker_size=20)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/oversubscription_across_{metric}.png")

if len(sys.argv) == 2:
    print("Passed a single path assuming it is a maximal batch size plotting")
    plot_maximal_batch_size(sys.argv[1])
else:
    plotting_type = sys.argv[1]
    if plotting_type == "single":
        plot_e2e(sys.argv[2:])
        plot_imbalance_and_oversubscription(sys.argv[2:])
        plot_average_speedup(sys.argv[2:])
        plot_overall_speedup(sys.argv[2:])
        plot_throughput(sys.argv[2:])
    elif plotting_type == "metric":
        metric = sys.argv[2]
        plot_speedup_across_metric(metric, sys.argv[3:])
        plot_imbalance_and_oversubscription_across_metric(metric, sys.argv[3:])
    else:
        print("No plotting type of that name")