import torch
import torch.nn.functional as F
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
import math
import random
from tqdm import tqdm
import deepspeed

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from flexible_dataset import FlexibleDataset
from modeling import Modeling
from parser import parse_arguments

args = parse_arguments()

def setup(rank):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    if args.system_name == "deepspeed-inference":
        deepspeed.init_distributed()
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.port
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        if args.path == "outputs":
            ROOT = f"{args.path}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        else:
            ROOT = args.path
        setup(rank)

        print("hello")
        model = Modeling(**vars(args))
        print("hello2")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache")

        flexible_dataset = FlexibleDataset(
            args.dataset, 
            tokenizer, 
            model, 
            seq_len=args.seq_len,
            num_samples=args.num_samples if args.num_samples != 0 else args.num_iters * args.batch_size * args.world
        )
        sampler = DistributedSampler(
            flexible_dataset, 
            num_replicas=dist.get_world_size(), 
            rank=rank, 
            shuffle=True, 
            seed=49
        )
        loader = DataLoader(
            flexible_dataset, 
            sampler=sampler, 
            batch_size=args.batch_size
        )

        run_standard_experiment(model, tokenizer, loader, f"{ROOT}/{rank}")
        if rank == 0:
            save_run_info(ROOT)

    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, tokenizer, loader, path):
    latencies = []

    if args.system_name == "deepspeed-inference":
        ds_engine = deepspeed.init_inference(
            model.model,
            dtype=torch.float,
            replace_with_kernel_inject=False,
            moe={
                "enabled": True,
                "ep_size": dist.get_world_size(), 
                "moe_experts": [model.num_experts],
            }
        )

        with torch.no_grad():
            for batch in tqdm(loader):
                start = time.time()
                batch = {k: v.cuda() for k, v in batch.items()}
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                )
                end = time.time()
                latencies.append(end-start)
    else:
        model.eval()
        model.cuda()
        with torch.no_grad():
            for batch in tqdm(loader):
                start = time.time()
                batch = {k: v.cuda() for k, v in batch.items()}
                model(batch)
                end = time.time()
                latencies.append(end-start)
    

    create_save_dir_if_not_exist(path)
    model.save_statistics(path)

    file_path = f"{path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["iteration", "latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow(
                {
                    "iteration": idx, 
                    "latency (s)": latency
                    }
            )

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
    run_info = vars(args).copy()

    if args.system_name == "deepspeed-inference" and dist.is_initialized():
        run_info["world_size"] = dist.get_world_size()

    with open(f"{path}/data.json", "w") as f:
        json.dump(run_info, f, indent=4)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)



    if args.system_name == "deepspeed-inference":
        # Assumes you execute with deepspeed command
        run_inference_workload(args.local_rank) 
    else:
        mp.spawn(run_inference_workload, nprocs=args.world_size, join=True)