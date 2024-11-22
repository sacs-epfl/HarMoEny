from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/experiment_2/"
data = Data(directory_path)

policies = ["deepspeed_policy", "harmony", "drop", "even_split"]
datasets = ["bookcorpus", "random", "wikitext", "wmt19"]
num_ranks = 8
num_moe_layers = 12

def create_workload_duration_policy_vs_dataset():
    df = []
    for policy in policies:
        for dataset in datasets:
            _df = data.load(f"{dataset}/{policy}/0/e2e.csv")
            df.append({
                "policy": policy,
                "dataset": dataset,
                "duration (s)": _df["latency (s)"].sum(axis=0)
            })
    save_pd(pd.DataFrame(df), "../data_processed/experiment_2/workload_duration_policy_vs_dataset.csv")

# Times are balanced proportional to the number of tokens
def create_moe_layer_latency_prob_num_tokens_policy_vs_dataset():
    for moe_layer_idx in range(num_moe_layers):
        df = []
        for policy in policies:
            for dataset in datasets:
                _df = data.load(f"{dataset}/{policy}/0/moe_layer-{moe_layer_idx}.csv")
                meta = data.read_meta(f"{dataset}/{policy}")
                df.append({
                    "policy": policy,
                    "dataset": dataset,
                    "latency (ms)": _df["latency (ms)"].mean() / meta["batch_size"]
                })
        save_pd(pd.DataFrame(df), f"../data_processed/experiment_2/layer_latency/layer-{moe_layer_idx}_avg_latency_prob_number_tokens_policy_vs_dataset.csv")

def create_imbalance_across_iter_and_average():
    for moe_layer_idx in range(num_moe_layers):
        df = []
        for dataset in datasets:
            dff = pd.DataFrame()
            for policy in policies:
                _df = None
                for rank in range(num_ranks):
                    __df = data.load(f"{dataset}/{policy}/{rank}/moe_layer-{moe_layer_idx}.csv")
                    __df = __df.rename(columns={"total number of tokens recv": rank})
                    if _df is None:
                        _df = __df[[rank]]
                    else:
                        _df = _df.merge(__df[[rank]], how="outer", on="iteration")
                _df["avg"] = _df.mean(axis=1)
                _df["max"] = _df.max(axis=1)
                _df["imbalance"] = ((_df["max"]-_df["avg"])/_df["avg"])*100
                _df["policy"] = policy
                _df = _df[["policy", "imbalance"]]
                dff = pd.concat([dff, _df])
                df.append({
                    "policy": policy,
                    "dataset": dataset,
                    "avg imbalance": _df["imbalance"].mean(axis=0)
                })
            save_pd(dff, f"../data_processed/experiment_2/imbalance/layer-{moe_layer_idx}_{dataset}_policy_imbalance_over_iters.csv")
        save_pd(pd.DataFrame(df), f"../data_processed/experiment_2/imbalance/layer-{moe_layer_idx}_avg_imbalance_policy_vs_dataset.csv")

def create_metric_across_iter_and_average():
    df = []
    for dataset in datasets:
        dff = pd.DataFrame()
        for policy in policies:
            _df = data.load(f"{dataset}/{policy}/stats.csv")
            meta = data.read_meta(f"{dataset}/{policy}")
            _df = _df[(_df["timestamp"] > meta["start"]) & (_df["timestamp"] < meta["end"])]
            _df['gpu_util'] = _df['gpu_util'].apply(ast.literal_eval)
            _df["avg_gpu_util"] = _df["gpu_util"].apply(lambda x: sum(x) / len(x))
            _df["gpu_mem_used"] = _df["gpu_mem_used"].apply(ast.literal_eval)
            _df["avg_gpu_mem_used"] = _df["gpu_mem_used"].apply(lambda x: sum(x) / len(x))
            _df["timestamp"] = _df["timestamp"] - _df["timestamp"].min()
            _df = _df.rename(columns={"timestamp": "seconds"})
            _df = _df.reset_index(drop=True)
            _df["policy"] = policy
            _df = _df[["policy", "seconds", "avg_gpu_util", "avg_gpu_mem_used", "cpu_util", "cpu_mem_used"]]
            dff = pd.concat([dff, _df])
            df.append({
                "policy": policy,
                "dataset": dataset,
                "avg_gpu_util": _df["avg_gpu_util"].mean(),
                "avg_gpu_mem_used": _df["avg_gpu_mem_used"].mean(),
                "cpu_util": _df["cpu_util"].mean(),
                "cpu_mem_used": _df["cpu_mem_used"].mean(),
            })
        save_pd(dff, f"../data_processed/experiment_2/metrics/{dataset}_policy_metrics_over_iters.csv")
    save_pd(pd.DataFrame(df), f"../data_processed/experiment_2/metrics/avg_metrics_policy_vs_dataset.csv")

if __name__ == "__main__":
    create_workload_duration_policy_vs_dataset()
    create_moe_layer_latency_prob_num_tokens_policy_vs_dataset()
    create_imbalance_across_iter_and_average()
    create_metric_across_iter_and_average()

