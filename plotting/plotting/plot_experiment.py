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

def plot_e2e(variable: str, dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)

        df[variable] = data[variable]

        frames.append(df)

    df = pd.concat(frames)
    df = df.sort_values([variable, "iteration"])
    df = df[df["iteration"] > 3]

    fig = px.line(df, x="iteration", y="latency (s)", color=variable)
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e.png")


def plot_imbalance_and_oversubscription(variable: str, dirs: [str]):
    final_frames = []
    for _dir in dirs:
        data = read_data(_dir)
        num_gpus = int(data["world"])

        frames = []
        for rank in range(num_gpus):
            if not os.path.exists(f"{_dir}/{rank}/moe_l0.csv"):
                print(f"{_dir} has no breakdown per layer. Skipping.")
                return 

            df = pd.read_csv(f"{_dir}/{rank}/moe_l0.csv")

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


        df_combined[variable] = data[variable]
        final_frames.append(df_combined[[variable, "iteration", "imbalance", "oversubscription"]])
    
    df = pd.concat(final_frames)
    df = df.sort_values([variable, "iteration"])
    df = df[df["iteration"] > 3]

    fig = px.line(df, x="iteration", y="imbalance", color=variable, labels={"iteration": "iteration", "imbalance": "Imbalance (relative %)"})
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/imbalance.png")


    fig = px.line(df, x="iteration", y="oversubscription", color=variable, labels={"iteration": "iteration", "oversubscription": "Oversubscription (relative %)"})
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    fig.write_image(f"{OUTPUT_DIR}/oversubscription.png")

def plot_average_speedup(comparison: str, dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)
        df = df.rename(columns={"latency (s)": data["scheduling_policy"]})
        frames.append(df)
    if len(frames) < 1:
        print("No data to work on, finishing plot_average_speedup")
        return
    comb_df = frames[0]
    for df in frames[1:]:
        comb_df = comb_df.merge(df, on="iteration", how="outer")

    comb_df = comb_df[comb_df["iteration"] > 3]

    columns = comb_df.columns.values[1:]
    if comparison not in columns:
        print("To obtain speedups you need designate a comparitor")
        return

    for col in columns:
        if col == comparison:
            continue
        comb_df[col] = comb_df[comparison] / comb_df[col]
    
    d_avg = []
    for col in columns:
        d_avg.append([comb_df[col].mean()])
    

    avg_df = pd.DataFrame(d_avg, index=columns, columns=["average speedup"])
    avg_df = avg_df.drop(index=comparison)
    avg_df = avg_df.sort_index()

    fig = px.bar(avg_df, y="average speedup", labels={"index": "scheduling policy"}, text="average speedup")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/average_speedup.png")

def plot_overall_e2e(variable: str, comparison: str, dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)
        df = df.rename(columns={"latency (s)": data[variable]})
        frames.append(df)
    if len(frames) < 1:
        print("No data to work on, finishing plot_overall_e2e")
        return
    comb_df = frames[0]
    for df in frames[1:]:
        comb_df = comb_df.merge(df, on="iteration", how="outer")
    comb_df = comb_df[comb_df["iteration"] > 3]
    print(comb_df)

    columns = comb_df.columns.values[1:]
    if comparison not in columns:
        print("To obtain speedups you need designate a comparitor")
        return
    
    df = comb_df.mean(axis=0) * (comb_df.count(axis=0)+3)
    df = df.drop(index="iteration")

    df = df.sort_index()

    df_speedup = df[comparison] / df

    plot_df = pd.DataFrame({
        variable: df.index,
        'workload_duration': df.values,
        'speedup': df_speedup.values,
        'label': [f"{value:.2f} ({speedup:.2f}x)" for value, speedup in zip(df, df_speedup)]
    })
    plot_df['label'] = plot_df['label'].astype(str)

    fig = px.bar(plot_df, 
                 x=variable, 
                 y='workload_duration', 
                 text='label',
                 labels={"scheduling_policy": "scheduling policy", 
                         "workload_duration": "workload duration (s)"})
    
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/overall_e2e.png")    

def plot_throughput(comparison: str, dirs: [str]):
    throughput = {"policy": [], "throughput": []}
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        df = df[df["iteration"] > 3]
        num_iters = len(df)
        df = df.sum(axis=0)
        data = read_data(_dir)
        num_samples = int(data["world_size"]) * int(data["batch_size"]) * num_iters

        throughput["policy"].append(data["scheduling_policy"])
        throughput["throughput"].append(num_samples / df["latency (s)"])
    
    df = pd.DataFrame(throughput)

    if len(df[df["policy"]==comparison]) == 0:
        print("To obtain speedups you need designate a comparitor")
        return

    comp = df[df["policy"]==comparison]["throughput"].iloc[0]
    df["labels"] = [f"{val:.2f} ({val/comp:.2f}x)" for val in df["throughput"].tolist()]

    fig = px.bar(df, x="policy", y="throughput", text="labels", 
        labels={"policy": "scheduling policy", "throughput": "throughput (reqs/s)"})
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
    fig.update_yaxes(rangemode="tozero")
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
        fig.update_yaxes(rangemode="tozero")
        update_fig_to_theme(fig)

        create_dir_if_needed()
        fig.write_image(f"{OUTPUT_DIR}/maximal_batch.png")


def plot_speedup_across_metric(variable: str, metric: str, dirs: [str]):
    values = []

    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        _len = len(df)
        df = df[df["iteration"] > 3]
        df = df.mean(axis=0)
        data = read_data(_dir)
        values.append({
            metric: data[metric],
            variable: data[variable],
            "time": df["latency (s)"] * _len
        })

    df = pd.DataFrame(values)

    df = df.sort_values(variable, axis=0)
    df["labels"] = [f"{val:.2f}" for val in df["time"].tolist()]
    fig = px.scatter(df, x=variable, y="time", color=metric, text="labels", labels={"time": "time (s)"})
    fig.update_traces(textposition="middle right", marker_size=20)
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e.png")

    # print(df)
    # exit(0)
    # metrics = df[metric].unique()
    # df['speedup'] = float('nan')
    # for m in metrics:
    #     deepspeed_value = df[(df["metric"] == m) & (df["policy"] == "deepspeed")]["time"].iloc[0]
    #     speedup_values = deepspeed_value / df[df["metric"] == m]["time"]
    #     df.loc[df["metric"] == m, "speedup"] = speedup_values
    
    # df = df.sort_values("policy", axis=0)
    # df["labels"] = [f"{val:.2f}" for val in df["speedup"].tolist()]
    # fig = px.scatter(df, x="policy", y="speedup", color="metric", text="labels", labels={"metric": metric})
    # fig.update_traces(textposition='middle right', marker_size=20)
    # fig.update_yaxes(rangemode="tozero")
    # update_fig_to_theme(fig)

    # create_dir_if_needed()
    # fig.write_image(f"{OUTPUT_DIR}/speedup_across_{metric}.png")

def plot_imbalance_and_oversubscription_across_metric(metric: str, dirs: [str]):
    values = []

    for _dir in dirs:
        data = read_data(_dir)

        interm2 = []

        for z in [0,1,2,3,4,5]: 
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
            "policy": data["scheduling_policy"],
            "imbalance": df["imbalance"].mean(axis=0),
            "oversubscription": df["oversubscription"].mean(axis=0)
        })

    df = pd.DataFrame(values)

    df = df.sort_values("policy", axis=0)

    df["labels_imbalance"] = [f"{val:.2f}" for val in df["imbalance"].tolist()]
    df["labels_oversubscription"] = [f"{val:.2f}" for val in df["oversubscription"].tolist()]
    
    fig = px.scatter(df, x="policy", y="imbalance", color="metric", text="labels_imbalance", labels={"metric": metric})
    fig.update_traces(textposition='middle right', marker_size=20)
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/imbalance_across_{metric}.png")

    fig = px.scatter(df, x="policy", y="oversubscription", color="metric", text="labels_oversubscription")
    fig.update_traces(textposition='middle right', marker_size=20)
    fig.update_yaxes(rangemode="tozero")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/oversubscription_across_{metric}.png")

if len(sys.argv) == 2:
    print("Passed a single path assuming it is a maximal batch size plotting")
    plot_maximal_batch_size(sys.argv[1])
else:
    plotting_type = sys.argv[1]
    if plotting_type == "single":
        variable = sys.argv[2]
        baseline = sys.argv[3]
        plot_e2e(variable, sys.argv[4:])
        #plot_imbalance_and_oversubscription(variable, sys.argv[4:])
        plot_overall_e2e(variable, baseline, sys.argv[4:])
        # Throughput a pretty irrelevant metric given e2e
        #plot_throughput(baseline, sys.argv[3:])
    elif plotting_type == "metric":
        variable = sys.argv[2]
        metric = sys.argv[3]
        plot_speedup_across_metric(variable, metric, sys.argv[4:])
        #plot_imbalance_and_oversubscription_across_metric(metric, sys.argv[4:])
    else:
        print("No plotting type of that scheduling_policy")