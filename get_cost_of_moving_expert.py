import os 
os.environ["HF_HOME"] = "/cache"

from transformers import AutoModel
import torch
import time
import csv
import copy
import torch.nn as nn
import math

model = AutoModel.from_pretrained("google/switch-base-8")

# Measures in ms

def cpu_gpu(expert):
    expert = expert.to("cpu")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    expert = expert.to("cuda:0")
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    return expert, elapsed_time

def gpu_cpu(expert):
    expert = expert.to("cuda:0")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    expert = expert.to("cpu")
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    return expert, elapsed_time

def gpu_gpu(expert, gpu_num):
    expert = expert.to("cuda:0")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    expert = expert.to(f"cuda:{gpu_num}")
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    return expert, elapsed_time

def gpu_cpu_gpu(expert, gpu_num):
    expert = expert.to("cuda:0")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    expert = expert.to("cpu")
    expert = expert.to(f"cuda:{gpu_num}")
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    return expert, elapsed_time

def create_random_expert():
    module = nn.ModuleDict({
        "wi": nn.Linear(768, 3072, bias=False),
        "wo": nn.Linear(3072, 768, bias=False),
        "dropout": nn.Dropout(p=0.1, inplace=False),
        "act": nn.ReLU()
    })
    nn.init.kaiming_uniform_(module['wi'].weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(module['wo'].weight, a=math.sqrt(5))
    return module

def run_an_experiment(num_iters, experiment, name, transfer_same=True):
    expert_to_move = model.encoder.block[1].layer[1].mlp.experts.expert_0 if transfer_same else create_random_expert()
    latencies = []
    for i in range(num_iters):
        if transfer_same is False:
            expert_to_move = create_random_expert()
        expert, dt = experiment(expert_to_move)
        latencies.append(dt)
        expert_to_move = expert 
    mean = sum(latencies) / len(latencies)
    variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
    _min = min(latencies)
    _max = max(latencies)
    return {
        "name": name,
        "mean": mean,
        "variance": variance,
        "min": _min,
        "max": _max,
        "raw values": latencies,
    }

def experiment():
    fieldnames = ["name", "mean", "variance", "min", "max", "raw values"]
    with open("outputs/moving_latencies_no_cache.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(run_an_experiment(100, cpu_gpu, "CPU to GPU:0", transfer_same=False))
        writer.writerow(run_an_experiment(100, gpu_cpu, "GPU:0 to CPU", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 1), "GPU:0 to GPU:1", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 2), "GPU:0 to GPU:2", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 3), "GPU:0 to GPU:3", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 4), "GPU:0 to GPU:4", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 5), "GPU:0 to GPU:5", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 6), "GPU:0 to GPU:6", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_gpu(e, 7), "GPU:0 to GPU:7", transfer_same=False))
        writer.writerow(run_an_experiment(100, lambda e: gpu_cpu_gpu(e, 1), "GPU:0 to CPU to GPU:1", transfer_same=False))

experiment()