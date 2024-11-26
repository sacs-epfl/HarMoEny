import pandas as pd
import sys
import json

if len(sys.argv) < 3:
    print("Please provide path of exflow solution and number of experts")
    exit(0)

df = pd.read_csv(sys.argv[1])
num_experts = int(sys.argv[2])
num_gpus = df["Node"].max() + 1
num_layers = len(df) // num_experts
arr = [[[] for _ in range(num_gpus)] for _ in range(num_layers)]
for i in range(num_layers):
    for j in range(num_experts):
        row = df[df["Expert"] == i * num_experts + j].iloc[0]
        arr[i][row["Node"]].append(j)

with open(f"placement/exp{num_experts}_gpu{num_gpus}.json", "w") as f:
    json.dump(arr, f)
