import sys
import pandas as pd 
from utils import save_pd

if len(sys.argv) < 2:
    print("Please provide path to do breakdown")

num_layers = 12
num_gpus = 8
path = sys.argv[1]
df = pd.DataFrame()
for i in range(num_layers):
    _df = pd.DataFrame()
    for j in range(num_gpus):
        __df = pd.read_csv(f"{path}/{j}/moe_layer-{i}.csv")
        __df = __df.loc[:, __df.columns.str.contains(r"\(ms\)", regex=True)]
        __df = __df.mean(axis=0)
        __df["rank"] = j

        _df = pd.concat([_df, __df.to_frame().T], ignore_index=True)

    _df["rank"] = _df["rank"].astype(int)
    max_comp = _df["comp latency (ms)"].max()
    _df["wait latency (ms)"] = max_comp - _df["comp latency (ms)"]
    _df["second transfer latency (ms)"] -= _df["wait latency (ms)"]
    columns_to_sum = _df.columns.difference(["rank", "latency (ms)"])
    _df["other latency (ms)"] = (_df["latency (ms)"] - _df[columns_to_sum].sum(axis=1)).clip(lower=0)
    _df["layer"] = i
    
    columns = _df.columns.values.tolist()
    columns.remove("layer")
    columns.insert(0, "layer")
    _df = _df[columns]

    df = pd.concat((df, _df))

save_pd(df, f"../data_processed/timebreakdown/{path.split('/')[-1]}.csv")