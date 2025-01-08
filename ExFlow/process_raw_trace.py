import pandas as pd
import numpy as np 
import ast
import sys 
import json 
import os 

if len(sys.argv) < 2:
    print("Please provide root path to conversion request")
    exit(0)

path = sys.argv[1]

if path[-1] == "/":
    path = path[:-1]

with open(f"{path}/data.json", "r") as f:
    meta = json.load(f)

num_layers = 24
num_ranks = meta["world_size"]
num_experts = meta["num_experts"]
num_tokens = meta["seq_len"] * meta["batch_size"] * num_ranks

token_layer_freq = np.zeros((num_tokens, num_layers), dtype=int)
for i in range(num_layers):
    t = 0
    for r in range(num_ranks):
        _df = pd.read_csv(f"{path}/{r}/moe_layer-{i}.csv")
        distr = _df["expert distribution"].apply(ast.literal_eval).iloc[0]
        for e in range(num_experts):
            for _ in range(distr[e]):
                token_layer_freq[t][i] = e
                t += 1

print(token_layer_freq)
np.save(f"trace/{os.path.splitext(os.path.basename(path))[0]}.npy", token_layer_freq)

# Verify
for j in range(num_layers):
    tmp = [0 for _ in range(128)]
    for i in range(num_tokens):
        tmp[token_layer_freq[i][j]] += 1
    print(sum(tmp))

        
