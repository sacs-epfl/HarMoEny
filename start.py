import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import sys
import os 
from datetime import datetime 
import time
import csv
import stat 
import json

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, SwitchTransformersEncoderModel

def setup(rank, world_size, port="12345"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def move_to_cuda_except_experts(model):
    for name, module in model.named_children():
        if name == 'experts':
            # We want to keep the experts on cpu
            continue
        elif list(module.children()):
            # If the module has children, recurse
            move_to_cuda_except_experts(module)
        else:
            # If it's a leaf module (no children) and not part of experts, move to CUDA
            module.cuda()

def run_inference_workload(rank, world_size, port, scheduling_policy):
    ROOT = f"outputs/{scheduling_policy}/{port}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    DIR = f"{ROOT}/{rank}"
    setup(rank, world_size, port)

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8", scheduling_policy=scheduling_policy)
    move_to_cuda_except_experts(model)
    model.expert_parallelise()

    datasets.enable_caching()
    dataset = load_dataset("bookcorpus/bookcorpus", split="train[:25000]", streaming=False, trust_remote_code=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=49)
    batch_size=250
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    latencies = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            start = time.time()
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            end = time.time()
            latencies.append(end-start)

    
    if not os.path.exists(DIR):
        os.makedirs(DIR, exist_ok=True)
        os.chmod(DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    
    model.expert_save_latencies(f"{DIR}")

    with open(f"{ROOT}/data.json", "w") as f:
        json.dump({
            "name": scheduling_policy
        }, f)

    file_path = f"{DIR}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["Iteration Number", "Latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({"Iteration Number": idx, "Latency (s)": latency})

    os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    cleanup()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python3 start.py num_gpus port_number scheduling_policy")
        exit(1)

    world_size = int(sys.argv[1])
    port = sys.argv[2]
    scheduling_policy = sys.argv[3]
    mp.spawn(run_inference_workload, args=(world_size, port, scheduling_policy), nprocs=world_size, join=True)