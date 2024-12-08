from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os
import ast

directory_path = "../data/stats/dataset_skew"
data = Data(directory_path)

df = []
num_layers = 6
num_ranks = 8
for dataset in data.get_children():
    skews = None
    for i in range(num_ranks):
        for j in range(num_layers):
            _df = data.load(f"{dataset}/{i}/moe_layer-{j}.csv")[["expert distribution"]]
            _df["expert distribution"] = _df["expert distribution"].apply(ast.literal_eval)
            _df["top_5"] = _df["expert distribution"].apply(lambda x: sorted(x, reverse=True)[:5])
            _df["tot"] = _df["expert distribution"].apply(sum)

            new_df = pd.DataFrame()
            for i in range(1, 6):
                new_df[f'top-{i} skew'] = (_df["top_5"].apply(lambda x: sum(x[:i])) / _df["tot"])

            if skews is None:
                skews = new_df
            else:
                skews = pd.concat([skews, new_df])

    skews = skews.mean(axis=0)
    df.append({
        "dataset": dataset,
        **skews.to_dict(),
    })

df = pd.DataFrame(df)
save_pd(df, "../data_processed/stats/dataset_skew/dataset_skew.csv")