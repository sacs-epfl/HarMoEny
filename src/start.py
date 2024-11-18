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
import argparse
import math
import random
from tqdm import tqdm
import deepspeed

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from flexible_dataset import FlexibleDataset
from modeling import Modeling

def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

# Argparse
parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)

parser.add_argument("-sys", "--system_name", choices=Modeling.available_systems(), default="harmony", type=str)
parser.add_argument("-d", "--dataset", default="sst2", type=str)
parser.add_argument("-ns", "--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("-bs", "--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("-sl", "--seq_len", default=120, type=int)
parser.add_argument("-m", "--model_name", default="google/switch-base-64", type=str, help="Huggingface model")
parser.add_argument("-nm", "--name_moe_layer", default="SwitchTransformersSparseMLP", type=str, help="class name of model MoELayers")
parser.add_argument("-nr", "--name_router", default="router", type=str, help="parameter name of router on MoE")
parser.add_argument("-ne", "--name_experts", default="experts", type=str, help="parameter name of router on MoE")
parser.add_argument("-nd", "--name_decoder", default="decoder", type=str, help="module name of model decoder")
parser.add_argument("-dc", "--dynamic_components", default=["wi", "wo"], type=list, help="parameter names of expert changing weights")
parser.add_argument("-pa", "--path", default="outputs", type=str, help="Specify where to save path")
parser.add_argument("-td", "--time_dense", default=False, type=str2bool, help="If you want to time the dense feed-forward")

args, remaining_argv = parser.parse_known_args()

if args.system_name != "deepspeed-inference":
    parser.add_argument("-w", "--world_size", default=torch.cuda.device_count(), type=int)
    parser.add_argument("-p", "--port", default="1234", type=str)

if args.system_name == "harmony":
    parser.add_argument("-sched", "--scheduling_policy", default="deepspeed", type=str)
    parser.add_argument("-cp", "--cache_policy", default="RAND", type=str)
    parser.add_argument("-ec", "--expert_cache_size", default=2, type=int)
    parser.add_argument("-eq", "--eq_tokens", default=150, type=int)

if args.system_name == "deepspeed-inference":
    parser.add_argument("--local_rank", default=0, type=int) 

args = parser.parse_args(remaining_argv, namespace=args)

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