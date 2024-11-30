import sys 
import pandas as pd 
import ast

if len(sys.argv) < 2:
    print("Please provide path to run")

path = sys.argv[1]

num_ranks = 8
num_layers = 6
highest_cap = 0
for l in range(num_layers):
    _df = None
    for r in range(num_ranks):
        __df = pd.read_csv(f"{path}/{r}/moe_layer-{l}.csv", index_col=0)
        __df = __df[["expert distribution"]]
        __df["expert distribution"] = __df["expert distribution"].apply(ast.literal_eval)
        __df["max"] = __df["expert distribution"].apply(max)
        __df["avg"] = __df["expert distribution"].apply(sum) / __df["expert distribution"].apply(len)
        __df["cap"] = __df["max"] / __df["avg"]
        cap = __df["cap"].max()
        if cap > highest_cap:
            highest_cap = cap

print(highest_cap)
