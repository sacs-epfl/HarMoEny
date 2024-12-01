import pandas as pd 
import sys
import ast

if len(sys.argv) < 2:
    print("Please provide path")

num_ranks = 8
num_layers = 6

_sum = 0
for i in range(num_ranks):
    for j in range(num_layers):
        df = pd.read_csv(f"{sys.argv[1]}/{i}/moe_layer-{j}.csv", index_col=0)
        df = df[["expert distribution"]]
        df["expert distribution"] = df["expert distribution"].apply(ast.literal_eval)
        df = pd.DataFrame(df["expert distribution"].tolist(), index=df.index)
        df["skew"] = df.max(axis=1) / df.sum(axis=1)
        _sum += df["skew"].mean(axis=0)

print(_sum / (num_ranks*num_layers))