import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import pynvml
import psutil
from threading import Thread
import sys
import os 
from datetime import datetime, timedelta
import time
import csv
import stat 
import json
import numpy as np
import pandas as pd
import signal
import math
import random
from tqdm import tqdm
from copy import deepcopy
import argparse

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset

from harmonymoe.utils import replace_moe_layer, get_moe_experts, get_moe_layers
from harmonymoe.moe_layer import MoEConfig, MoELayer

def str2bool(s):
        return s.lower() in ["yes", "y", "true", "t"]

parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)
parser.add_argument("--dataset", default="sst2", type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size per GPU")
parser.add_argument("--start_batch_size", default=1, type=int)
parser.add_argument("--seq_len", default=120, type=int)
parser.add_argument("--model_name", default="google/switch-base-64", type=str, help="Huggingface model")
parser.add_argument("--type_moe_parent", default="SwitchTransformersLayerFF", type=str, help="class name of model MoE Layer parent")
parser.add_argument("--type_moe", default="SwitchTransformersSparseMLP", type=str, help="class name of model MoE Layers")
parser.add_argument("--name_router", default="router", type=str, help="parameter name of router on MoE")
parser.add_argument("--name_experts", default="experts", type=str, help="parameter name of router on MoE")
parser.add_argument("--name_decoder", default="decoder", type=str, help="module name of model decoder")
parser.add_argument("--dynamic_components", default=["wi", "wo"], type=list, help="parameter names of expert changing weights")
parser.add_argument("--d_model", default=768, type=int, help="Dimension of model hidden states")
parser.add_argument("--scheduling_policy", default="deepspeed", type=str)
parser.add_argument("--cache_policy", default="RAND", type=str)
parser.add_argument("--expert_cache_size", default=2, type=int)
parser.add_argument("--eq_tokens", default=150, type=int)
parser.add_argument("--world_size", default=torch.cuda.device_count(), type=int)
parser.add_argument("--port", default="1234", type=str)
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("--path", default="outputs/harmony", type=str)
parser.add_argument("--expert_placement", default=None, type=str)
args = parser.parse_args()

############# GLOBAL AFFAIRS ################
pynvml.nvmlInit()
#############################################

def get_size(obj):
    obj.cuda()
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.memory_allocated(device=0)
    obj.cpu()
    torch.cuda.synchronize()
    return memory_allocated

def setup(rank, timeout=timedelta(minutes=30)):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size, timeout=timeout)

    torch.cuda.set_device(rank)

def check_inference(model, batch):
    try:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            model(**batch)
        success = True
        print("SUCCESS")
    except Exception as e:
        print(f"/////////////BIG ERROR: {str(e)}")
        success = False
        print("FAIL")
    finally:
        torch.cuda.empty_cache()
        return success

def compute_batch_size(model, dataset, sampler):
    batch = args.start_batch_size
    incr = 20

    while True:
        test = batch + incr
        print(f"TESTING {test}")
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=test,
        )
        if check_inference(model, next(iter(dataloader))):
            batch = test
        else:
            return batch

def find_batch_size(rank, model, batch_sizes):
    mp.current_process().name = f'Worker-{rank}'
    setup(rank, timeout=timedelta(seconds=60))
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "2"
    #os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"

    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache")

    flexible_dataset = FlexibleDataset(
        args.dataset, 
        tokenizer, 
        model, 
        seq_len=args.seq_len,
        num_samples=args.num_samples
    )
    sampler = DistributedSampler(
        flexible_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True, 
        seed=49
    )

    max_batch_size = compute_batch_size(model, flexible_dataset, sampler)
    print(f"THE MAX: {max_batch_size}")
    batch_sizes.append(max_batch_size)

def run_inference_workload(rank, model, batch=args.batch_size):
    try:
        args.batch_size = batch

        mp.current_process().name = f'Worker-{rank}'
        setup(rank)

        model.cuda()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache")

        flexible_dataset = FlexibleDataset(
            args.dataset, 
            tokenizer, 
            model, 
            seq_len=args.seq_len,
            num_samples=args.num_samples
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
            batch_size=batch,
        )

        path = f"{args.path}/{rank}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        latencies, run_start, run_end = run_standard_experiment(model, loader)
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

        for l in get_moe_layers(model):
            file_path = f"{path}/moe_layer-{l.layer_idx}.csv"
            stats = l.get_statistics()[args.warmup_rounds:]

            with open(file_path, "w") as f:
                fieldnames = ["iteration", "total number of tokens sent", "total number of tokens recv", "latency (ms)", "metadata latency (ms)", "comp latency (ms)", "expert distribution"]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(len(stats)):
                    dic = {
                        "iteration": i,
                        **stats[i]
                    }
                
                    writer.writerow(dic)   

        if rank == 0:
            run_info = vars(args).copy()
            with open(f"{args.path}/data.json", "w") as f:
                json.dump({ "start": run_start, "end": run_end, **run_info}, f, indent=4)


    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, loader):
    latencies = []

    with torch.no_grad():
        # WARMUP
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            model(**batch)
            itr += 1
            if itr == args.warmup_rounds:
                break

        run_start = time.time()
        # RUN ACTUAL EXPERIMENT
        for batch in tqdm(loader):
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            model(**batch)
            end = time.time()
            latencies.append(end-start)
        run_end = time.time()
    
    return latencies, run_start, run_end

def cleanup():
    dist.destroy_process_group()

def fetch_metrics(stop_event, output_list):
    handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(args.world_size)]

    while not stop_event.is_set():
        output_list.append({
            "timestamp": time.time(),
            "gpu_util": [pynvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in handles],
            "gpu_mem_used": [pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles],
            "cpu_util": psutil.cpu_percent(interval=None),
            "cpu_mem_used": psutil.virtual_memory().used,
        })

        time.sleep(1)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    model = AutoModel.from_pretrained(args.model_name, cache_dir="/cache")
    experts = get_moe_experts(model, args.type_moe, args.name_experts)
    experts.share_memory()
    config = MoEConfig(
        scheduling_policy=args.scheduling_policy,
        cache_policy=args.cache_policy,
        expert_cache_size=args.expert_cache_size,
        dynamic_components=args.dynamic_components,
        eq_tokens=args.eq_tokens,
        d_model=args.d_model,
        world_size=args.world_size,
        expert_placement=args.expert_placement,
    )
    replace_moe_layer(
        model, 
        args.type_moe_parent,
        args.type_moe, 
        args.name_router, 
        experts,
        config,
    )
    
    if args.batch_size == None:
        manager = mp.Manager()
        batch_sizes = manager.list()

        processes = []
        mp.set_start_method("spawn", force=True)
        for i in range(args.world_size):
            p = mp.Process(target=find_batch_size, args=(i, model, batch_sizes))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        args.batch_size = min(batch_sizes)

    print(f"NEW BATCH SIZE: {args.batch_size}")

    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics))
    metric_thread.start()

    processes = []
    mp.set_start_method("spawn", force=True)
    for i in range(args.world_size):
        p = mp.Process(target=run_inference_workload, args=(i, model), kwargs=dict(batch=args.batch_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    stop_event.set()
    metric_thread.join()

    df = pd.DataFrame(metrics)
    df.to_csv(f"{args.path}/stats.csv")

    print("All done :)")

    pynvml.nvmlShutdown()