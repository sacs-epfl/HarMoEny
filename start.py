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
import numpy as np
import signal

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, SwitchTransformersEncoderModel

SEQ_LEN = 120
NUM_ITERS = 100

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


class FlexibleDataset(Dataset):
    def __init__(self, dataset_option, tokenizer, world_size, random_seed=32, batch_size=60):
        self.tokenizer = tokenizer
        self.max_length = SEQ_LEN
        self.dataset_option = dataset_option
        self.dataset_size = NUM_ITERS * batch_size * world_size
        torch.manual_seed(random_seed)

        if dataset_option == "bookcorpus":
            self.dataset = load_dataset("bookcorpus/bookcorpus", split=f"train[:{self.dataset_size}]", streaming=False, trust_remote_code=True, cache_dir="/cache")
        elif dataset_option == "wikitext":
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{self.dataset_size}]", streaming=False, cache_dir="/cache")
        elif dataset_option == "sst2":
            self.dataset = load_dataset("glue", "sst2", split=f"train[:{self.dataset_size}]", streaming=False, cache_dir="/cache")
        elif dataset_option == "random":
            self.vocab_size = len(tokenizer)
        else:
            raise ValueError("Invalid dataset option")

        self.dataset_size = min(self.dataset_size, len(self.dataset))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset_option != "random":
            if self.dataset_option == "bookcorpus" or self.dataset_option == "wikitext":
                text = self.dataset[idx]["text"]
            elif self.dataset_option == "sst2":
                text = self.dataset[idx]["sentence"]
            encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in encoded.items()}
        else:  # random dataset            
            # Generate random token IDs
            input_ids = torch.randint(0, self.vocab_size, (self.max_length,), dtype=torch.int64)            
            
            # Create attention mask
            attention_mask = torch.ones(self.max_length, dtype=torch.int64)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

def run_inference_workload(rank, world_size, port, scheduling_policy, dataset, experiment, batch_size):
    try:
        mp.current_process().name = f'Worker-{rank}'
        ROOT = f"outputs/{scheduling_policy}/{port}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        DIR = f"{ROOT}/{rank}"
        setup(rank, world_size, port)

        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir="/cache")
        model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8", scheduling_policy=scheduling_policy, cache_dir="/cache")
        move_to_cuda_except_experts(model)
        model.expert_parallelise()

        datasets.enable_caching()
        flexible_dataset = FlexibleDataset(dataset, tokenizer, world_size, batch_size=batch_size)
        sampler = DistributedSampler(flexible_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=49)
        loader = DataLoader(flexible_dataset, sampler=sampler, batch_size=batch_size)
        
        model.eval()

        if experiment == "throughput":
            run_throughput_experiment(model, loader, flexible_dataset, sampler, DIR)
        else:
            run_standard_experiment(model, loader, DIR)
            save_run_info(scheduling_policy, batch_size, world_size, dataset, ROOT)

    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, loader, path):
    latencies = []
    
    with torch.no_grad():
        for batch in loader:
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            end = time.time()
            latencies.append(end-start)
    
    create_save_dir_if_not_exist(path)
    
    model.expert_save_latencies(f"{path}")

    file_path = f"{path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["Iteration Number", "Latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({"Iteration Number": idx, "Latency (s)": latency})

def run_throughput_experiment(model, loader, dataset, sampler, path):
    warmup(model, loader)

    print(f"Time for batch size: {run_x_times_and_get_average(model, loader)}")


    #CUTOFFS = [1.0, 2.0, 3.0]
    # CUTOFFS = [1.0]

    # warmup(model, loader)

    # results = []
    # for cutoff in CUTOFFS:
    #     left = 0
    #     right = 100
    #     while True: # Get the right bound
    #         avg = run_x_times_and_get_average(model, dataset, sampler, batch_size=right)
    #         if avg > cutoff:
    #             break
    #         else:
    #             left = right
    #             right *= 2
    #     while True: # Shrink the right bound
    #         if right == left:
    #                 break
    #         test_size = (right + left) // 2
    #         test_size = max(test_size, left)
    #         avg = run_x_times_and_get_average(model, dataset, sampler, batch_size=test_size)
    #         if avg > cutoff:
    #             right = test_size
    #         else:
    #             left = test_size
    #     results.append(right)
    
    # create_save_dir_if_not_exist(path)

    # file_path = f"{path}/throughput.csv"
    # with open(file_path, "w") as f:
    #     writer = csv.DictWriter(f, fieldnames=CUTOFFS)
    #     writer.writeheader()
    #     writer.writerow(dict(zip(CUTOFFS, results)))


def warmup(model, loader):
    NUM_WARMUP_ROUNDS = 3

    itr = 0
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        itr += 1
        if itr == NUM_WARMUP_ROUNDS:
            break

def run_x_times_and_get_average(model, loader, times=3):
    tot_run_time = 0
    itr = 0
    with torch.no_grad():
        for batch in loader:
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            end = time.time()
            tot_run_time += end-start 
            itr += 1
            if itr == times:
                break
    
    return tot_run_time / times

def create_save_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_run_info(scheduling_policy, batch_size, world_size, dataset, path):
     with open(f"{path}/data.json", "w") as f:
        json.dump({
            "name": scheduling_policy,
            "batch_size": batch_size,
            "world_size": world_size,
            "dataset": dataset,
        }, f)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("usage: python3 start.py num_gpus port_number scheduling_policy dataset experiment [batch_size]")
        exit(1)
    world_size = int(sys.argv[1])
    port = sys.argv[2]
    scheduling_policy = sys.argv[3]
    dataset = sys.argv[4]
    experiment = sys.argv[5]
    batch_size = 250
    if len(sys.argv) > 6:
        batch_size = int(sys.argv[6])

    signal.signal(signal.SIGINT, signal_handler)
    mp.spawn(run_inference_workload, args=(world_size, port, scheduling_policy, dataset, experiment, batch_size), nprocs=world_size, join=True)
    
