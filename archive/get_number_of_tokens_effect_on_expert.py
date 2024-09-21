import os 
os.environ["HF_HOME"] = "/cache"

from transformers import AutoModel
import torch
import time
import csv

model = AutoModel.from_pretrained("google/switch-base-8")
expert = model.encoder.block[1].layer[1].mlp.experts.expert_0.to("cuda:0")

HIDDEN_SIZE = 768
EXPERIMENTATION_NUM_ITERS = 100

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def get_execution_time_for_number_of_tokens(num_tokens):
    tensor = torch.rand((1, num_tokens, HIDDEN_SIZE)).to("cuda:0")
    start.record()
    expert(tensor)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    return elapsed_time

def run_experiment(num_tokens):
    latencies = []
    get_execution_time_for_number_of_tokens(num_tokens)
    for _ in range(EXPERIMENTATION_NUM_ITERS):
        latencies.append(get_execution_time_for_number_of_tokens(num_tokens))
    return sum(latencies) / len(latencies)

NUM_TOKENS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4095, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

with open("outputs/latency_expert_over_num_tokens.csv", "w") as f:
    fieldnames = ["Num Tokens", "Latency (ms)"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for num_tokens in NUM_TOKENS_LIST:
        writer.writerow({"Num Tokens": num_tokens, "Latency (ms)": run_experiment(num_tokens)})