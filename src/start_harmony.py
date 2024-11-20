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
from datetime import datetime 
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

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
from modeling import Modeling
from parser import ArgParse

from harmonymoe.utils import replace_moe_layer, get_moe_experts, get_moe_layers
from harmonymoe.moe_layer import MoEConfig, MoELayer

parser = ArgParse()
parser.add_argument("--scheduling_policy", default="deepspeed", type=str)
parser.add_argument("--cache_policy", default="RAND", type=str)
parser.add_argument("--expert_cache_size", default=2, type=int)
parser.add_argument("--eq_tokens", default=150, type=int)
parser.add_argument("--world_size", default=torch.cuda.device_count(), type=int)
parser.add_argument("--port", default="1234", type=str)
args = parser.parse_arguments()


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

def setup(rank):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.cuda.set_device(rank)

def run_inference_workload(rank, model, ROOT):
    try:
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
            num_samples=args.num_samples if args.num_samples != 0 else args.num_iters * args.batch_size * args.world_size
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

        latencies, run_start, run_end = run_standard_experiment(model, loader)
        path = f"{ROOT}/{rank}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
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
            l.save_statistics(path)
        if rank == 0:
            run_info = vars(args).copy()
            with open(f"{ROOT}/data.json", "w") as f:
                json.dump({ "start": run_start, "end": run_end, **run_info}, f, indent=4)


    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, loader):
    latencies = []

    with torch.no_grad():
        # WARMUP
        NUM_WARMUP_ROUNDS = 3
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            model(**batch)
            itr += 1
            if itr == NUM_WARMUP_ROUNDS:
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

def fetch_metrics(stop_event, output_list, ROOT):
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
    )
    replace_moe_layer(
        model, 
        args.type_moe_parent,
        args.type_moe, 
        args.name_router, 
        experts,
        config,
    )

    if not args.path:
        ROOT = f"outputs/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    else:
        ROOT = args.path

    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics, ROOT))
    metric_thread.start()

    processes = []
    mp.set_start_method('spawn', force=True)
    for i in range(args.world_size):
        p = mp.Process(target=run_inference_workload, args=(i, model, ROOT))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    stop_event.set()
    metric_thread.join()

    df = pd.DataFrame(metrics)
    df.to_csv(f"{ROOT}/stats.csv")

    print("All done :)")

    pynvml.nvmlShutdown()