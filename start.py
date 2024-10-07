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

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, SwitchTransformersEncoderModel, SwitchTransformersForConditionalGeneration

def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

# Argparse
parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)

parser.add_argument("-sl", "--seq_len", default=120, type=int)
parser.add_argument("-ni", "--num_iters", default=0, type=int)
parser.add_argument("-ns", "--num_samples", default=0, type=int, help="Number of samples per GPU")
parser.add_argument("-w", "--world", default=torch.cuda.device_count(), type=int)
parser.add_argument("-p", "--port", default="1234", type=str)
parser.add_argument("-s", "--schedule", default="deepspeed", type=str)
parser.add_argument("-d", "--dataset", default="sst2", type=str)
parser.add_argument("-bs", "--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("-x", "--experiment", default="standard", type=str)
parser.add_argument("-r", "--enable_rebalancing", default=False, type=str2bool)
parser.add_argument("-rf", "--rebalancing_frequency", default=15, type=int)
parser.add_argument("-me", "--max_loaded_experts", default=2, type=int)
parser.add_argument("-e", "--num_experts", default=8, type=int)
parser.add_argument("-mt", "--model_type", default="encoder-decoder", type=str)
parser.add_argument("-pa", "--path", default="outputs", type=str, help="Specify where to save path")

args = parser.parse_args()

# Max loaded experts must be greater than or equal to EP size
if args.max_loaded_experts < math.ceil(args.num_experts / args.world):
    print("The max loaded experts must be greater than the expert parallel size")
    exit(1)

if args.num_experts not in [8, 16, 32, 64, 128, 256]:
    print(f"There is no model with {args.num_experts} experts")
    exit(1)

if args.num_iters == 0 and args.num_samples == 0:
    print("You must either specify --num_iters or --num_samples")
    exit(1)

if args.num_iters != 0:
    DESIRED_DATASET_SIZE = args.num_iters * args.batch_size * args.world
else:
    DESIRED_DATASET_SIZE = args.num_samples * args.world

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
    def __init__(self, tokenizer, model, random_seed=32):
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.max_length = args.seq_len
        self.dataset_option = args.dataset
        torch.manual_seed(random_seed)

        if self.dataset_option == "bookcorpus":
            self.dataset = load_dataset("bookcorpus/bookcorpus", split=f"train[:{DESIRED_DATASET_SIZE}]", streaming=False, trust_remote_code=True, cache_dir="/cache")
        elif self.dataset_option == "wikitext":
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{DESIRED_DATASET_SIZE}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "sst2":
            self.dataset = load_dataset("glue", "sst2", split=f"train[:{DESIRED_DATASET_SIZE}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "wmt19":
            self.dataset = load_dataset("wmt/wmt19", "en-de", split=f"train[:{DESIRED_DATASET_SIZE}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "random":
            pass
        else:
            raise ValueError("Invalid dataset option")

        if self.dataset_option != "random":
            self.dataset_size = len(self.dataset)
        else:
            self.dataset_size = DESIRED_DATASET_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset_option == "bookcorpus" or self.dataset_option == "wikitext":
            encoder = "summarize: " + self.dataset[idx]["text"]
        elif self.dataset_option == "sst2":
            encoder = "summarize: " + self.dataset[idx]["sentence"]
        elif self.dataset_option == "wmt19":
            encoder = "translate English to German: " + self.dataset[idx]["translation"]["en"]
        elif self.dataset_option == "random":
            encoder = ["summarize:"]
            vocab_size = self.tokenizer.vocab_size

            for _ in range(args.seq_len):
                # Add a random token to the array
                random_token_id = random.randint(0, vocab_size-1)
                random_token = self.tokenizer.decode(random_token_id)
                encoder.append(random_token)

            encoder = " ".join(encoder)

        if args.batch_size == 1:
            encoder_tokenized = self.tokenizer(encoder, truncation=True, max_length=self.max_length, return_tensors="pt")
        else:
            encoder_tokenized = self.tokenizer(encoder, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        
        dic = {
            "input_ids": encoder_tokenized["input_ids"].squeeze(0),
            "attention_mask": encoder_tokenized["attention_mask"].squeeze(0),
        }

        if args.model_type == "encoder-decoder":
            dic["decoder_input_ids"] = torch.tensor([self.model_config.pad_token_id])

        return dic

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        ROOT = f"{args.path}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        setup(rank)

        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir="/cache")
        
        if args.model_type == "encoder-decoder":
            _class = SwitchTransformersForConditionalGeneration
        elif args.model_type == "encoder":
            _class = SwitchTransformersEncoderModel
        else:
            print("That model type is not yet implemented!")
            exit(1)
        
        model = _class.from_pretrained(
            f"google/switch-base-{args.num_experts}", 
            scheduling_policy=args.schedule, 
            enable_rebalancing=args.enable_rebalancing, 
            rebalancing_frequency=args.rebalancing_frequency,
            max_loaded_experts=args.max_loaded_experts,
            cache_dir="/cache")

        move_to_cuda_except_experts(model)
        model.expert_parallelise()

        datasets.enable_caching()
        flexible_dataset = FlexibleDataset(tokenizer, model)
        sampler = DistributedSampler(flexible_dataset, num_replicas=args.world, rank=rank, shuffle=True, seed=49)
        loader = DataLoader(flexible_dataset, sampler=sampler, batch_size=args.batch_size)

        model.eval()

        if args.experiment == "standard":
            run_standard_experiment(model, tokenizer, loader, f"{ROOT}/{rank}")
            save_run_info(ROOT)
        else:
            print(f"That experiment, {args.experiment}, is not yet implemented")
            exit(1)

    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, tokenizer, loader, path):
    latencies = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            if args.model_type == "encoder-decoder":
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_token_ids = torch.argmax(probs, dim=-1)
                predicted_token = [tokenizer.decode(pred_id, skip_special_tokens=True) for pred_id in predicted_token_ids]
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
            "seq_len": args.seq_len,
            "num_iters": args.num_iters,
            "num_samples": args.num_samples,
            "port": args.port,
            "experiment": args.experiment,
            "enable_rebalancing": args.enable_rebalancing,
            "max_loaded_experts": args.max_loaded_experts,
            "num_experts": args.num_experts,
            "model_type": args.model_type
        }, f)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    mp.spawn(run_inference_workload, nprocs=args.world, join=True)
    
