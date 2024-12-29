import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import os 
from datetime import datetime 
import time
import csv
import json
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
from utils import TimedModule, get_timing_modules

parser = argparse.ArgumentParser(
    prog="Run inference on t5",
)
parser.add_argument("--dataset", default="sst2", type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("--seq_len", default=120, type=int)
parser.add_argument("--path", default=None, type=str, help="Specify where to save path")
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("--cache-dir", default="/cache", type=str, help="The cache dir for data and models")
args = parser.parse_args()

def run_inference_workload():
    model = AutoModel.from_pretrained("google-t5/t5-base", cache_dir=args.cache_dir)

    def add_timing_model(model, idx):
        if type(model).__name__ == "T5LayerFF":
            for child_name, child in model.named_children():
                if type(child).__name__ == "T5DenseActDense":
                    setattr(model, child_name, TimedModule(child, idx=idx[0]))
                    idx[0] += 1
        else:
            for child in model.children():
                add_timing_model(child, idx)
    
    add_timing_model(model, [0])

    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", cache_dir=args.cache_dir)

    flexible_dataset = FlexibleDataset(
        args.dataset, 
        tokenizer, 
        model, 
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        flexible_dataset, 
        batch_size=args.batch_size
    )

    latencies, run_start, run_end = run_standard_experiment(model, loader)

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    ############# E2E #######################
    file_path = f"{args.path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["iteration", "latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({
                "iteration": idx, 
                "latency (s)": latency,
            })
    
    ############# LAYER #######################
    for timing_module in get_timing_modules([], model):
        latencies = timing_module.get_latencies()[args.warmup_rounds:]
        file_path = f"{args.path}/layer-{timing_module.idx}.csv"
        with open(file_path, "w") as f:
            fieldnames = ["iteration", "latency (ms)"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, latency in enumerate(latencies):
                writer.writerow({
                    "iteration": idx,
                    "latency (ms)": latency,
                })
    
    ############# META #######################
    run_info = vars(args).copy()
    with open(f"{args.path}/data.json", "w") as f:
        json.dump({ "start": run_start, "end": run_end, **run_info}, f, indent=4)

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

if __name__ == "__main__":
    torch.cuda.set_device(0)
    run_inference_workload()