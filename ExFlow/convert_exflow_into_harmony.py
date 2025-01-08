import pandas as pd
import sys
import json

if len(sys.argv) < 3:
    print("Please provide path of exflow solution and number of experts, you can optionally specify desired number of gpus")
    exit(0)

df = pd.read_csv(sys.argv[1])
num_experts = int(sys.argv[2])
if len(sys.argv) > 3:
    desired_num_gpus = int(sys.argv[3])
else:
    desired_num_gpus = num_experts
desired_ep_size = num_experts // desired_num_gpus
num_gpus = df["Node"].max() + 1
num_layers = len(df) // num_experts
arr = [[[] for _ in range(desired_num_gpus)] for _ in range(num_layers)]
for i in range(num_layers):
    for j in range(num_experts):
        row = df[df["Expert"] == i * num_experts + j].iloc[0]
        arr[i][row["Node"]].append(j)
    for j in range(desired_num_gpus):
        while len(arr[i][j]) > desired_ep_size:
            # Find gpu that is not full
            for r in range(j+1, desired_num_gpus):
                if len(arr[i][r]) < desired_ep_size:
                    arr[i][r].append(arr[i][j].pop())
                    break

with open(f"placement/exp{num_experts}_gpu{desired_num_gpus}.json", "w") as f:
    json.dump(arr, f)
