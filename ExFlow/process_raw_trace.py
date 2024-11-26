import pandas as pd
import numpy as np 
import ast
import sys 
import json 

if len(sys.argv) < 2:
    print("Please provide root path to conversion request")
    exit(0)

path = sys.argv[1]
with open(f"{path}/data.json", "r") as f:
    meta = json.load(f)

num_layers = 6 
num_ranks = meta["world_size"]
num_experts = int(meta["model_name"].split("-")[-1])
num_tokens = 5100 * num_ranks

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

np.save(f"trace/{num_experts}.npy", token_layer_freq)

# Verify
for j in range(num_layers):
    tmp = [0 for _ in range(128)]
    for i in range(num_tokens):
        tmp[token_layer_freq[i][j]] += 1
    print(sum(tmp))

        
