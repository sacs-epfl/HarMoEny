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
import argparse

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, SwitchTransformersEncoderModel

def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

# Argparse
parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)

parser.add_argument("-sl", "--seq_len", default=120, type=int)
parser.add_argument("-ni", "--num_iters", default=100, type=int)
parser.add_argument("-w", "--world", default=torch.cuda.device_count(), type=int)
parser.add_argument("-p", "--port", default="1234", type=str)
parser.add_argument("-s", "--schedule", default="deepspeed", type=str)
parser.add_argument("-d", "--dataset", default="sst2", type=str)
parser.add_argument("-bs", "--batch_size", default=250, type=int)
parser.add_argument("-e", "--experiment", default="standard", type=str)
parser.add_argument("-r", "--enable_rebalancing", default=True, type=str2bool)
parser.add_argument("-rf", "--rebalancing_frequency", default=15, type=int)
parser.add_argument("-me", "--max_loaded_experts", default=2, type=int)
parser.add_argument("-e", "--number_experts", default=8, type=int)

args = parser.parse_args()

# Max loaded experts must be greater than or equal to EP size
if args.max_loaded_experts < ceil(args.number_experts / args.world):
    print("The max loaded experts must be greater than the expert parallel size")
    exit(1)

if args.number_experts not in [8, 16, 32, 64, 128, 256]:
    print(f"There is no model with {args.number_experts} experts")
    exit(1)

def setup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    dist.init_process_group("nccl", rank=rank, world_size=args.world)
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
    def __init__(self, tokenizer, random_seed=32):
        self.tokenizer = tokenizer
        self.max_length = args.seq_len
        self.dataset_option = args.dataset
        self.dataset_size = args.num_iters * args.batch_size * args.world
        torch.manual_seed(random_seed)

        if self.dataset_option == "bookcorpus":
            self.dataset = load_dataset("bookcorpus/bookcorpus", split=f"train[:{self.dataset_size}]", streaming=False, trust_remote_code=True, cache_dir="/cache")
        elif self.dataset_option == "wikitext":
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{self.dataset_size}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "sst2":
            self.dataset = load_dataset("glue", "sst2", split=f"train[:{self.dataset_size}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "random":
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

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        ROOT = f"outputs/{args.schedule}/{args.port}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        DIR = f"{ROOT}/{rank}"
        setup(rank)

        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir="/cache")
        model = SwitchTransformersEncoderModel.from_pretrained(
            f"google/switch-base-{args.number_experts}", 
            scheduling_policy=args.schedule, 
            enable_rebalancing=args.enable_rebalancing, 
            rebalancing_frequency=args.rebalancing_frequency,
            max_loaded_experts=args.max_loaded_experts,
            cache_dir="/cache")
        move_to_cuda_except_experts(model)
        model.expert_parallelise()

        datasets.enable_caching()
        flexible_dataset = FlexibleDataset(tokenizer)
        sampler = DistributedSampler(flexible_dataset, num_replicas=args.world, rank=rank, shuffle=True, seed=49)
        loader = DataLoader(flexible_dataset, sampler=sampler, batch_size=args.batch_size)
        
        model.eval()

        if args.experiment == "throughput":
            run_throughput_experiment(model, loader, flexible_dataset, sampler, DIR)
        elif args.experiment == "standard":
            run_standard_experiment(model, loader, DIR)
            save_run_info(ROOT)
        else:
            print(f"That experiment, {args.experiment}, is not yet implemented")
            exit(1)

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

# def run_throughput_experiment(model, loader, dataset, sampler, path):
#     warmup(model, loader)

#     print(f"Time for batch size: {run_x_times_and_get_average(model, loader)}")


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

def save_run_info(path):
     with open(f"{path}/data.json", "w") as f:
        json.dump({
            "name": args.schedule,
            "batch_size": args.batch_size,
            "world_size": args.world,
            "dataset": args.dataset,
        }, f)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    # if len(sys.argv) < 6:
    #     print("usage: python3 start.py num_gpus port_number scheduling_policy dataset experiment [batch_size]")
    #     exit(1)
    # world_size = int(sys.argv[1])
    # port = sys.argv[2]
    # scheduling_policy = sys.argv[3]
    # dataset = sys.argv[4]
    # experiment = sys.argv[5]
    # batch_size = 250
    # if len(sys.argv) > 6:
    #     batch_size = int(sys.argv[6])

    signal.signal(signal.SIGINT, signal_handler)
    mp.spawn(run_inference_workload, nprocs=args.world, join=True)
    
