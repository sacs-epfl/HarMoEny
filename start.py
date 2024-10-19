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

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from testing.flexible_dataset import FlexibleDataset
from testing.modelling import Modelling

def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

# Argparse
parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)

parser.add_argument("-sl", "--seq_len", default=120, type=int)
parser.add_argument("-ni", "--num_iters", default=0, type=int)
parser.add_argument("-ns", "--num_samples", default=0, type=int, help="Number of total samples")
parser.add_argument("-w", "--world", default=torch.cuda.device_count(), type=int)
parser.add_argument("-p", "--port", default="1234", type=str)
parser.add_argument("-s", "--scheduling_policy", default="deepspeed", type=str)
parser.add_argument("-cp", "--cache_policy", default="RAND", type=str)
parser.add_argument("-d", "--dataset", default="sst2", type=str)
parser.add_argument("-bs", "--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("-x", "--experiment", default="standard", type=str)
parser.add_argument("-ec", "--expert_cache_size", default=2, type=int)
# parser.add_argument("-e", "--num_experts", default=8, type=int)
# parser.add_argument("-mt", "--model_type", default="encoder-decoder", type=str)
parser.add_argument("-pa", "--path", default="outputs", type=str, help="Specify where to save path")
parser.add_argument("-eq", "--eq_tokens", default=150, type=int)
parser.add_argument("-dm", "--d_model", default=768, type=int)
parser.add_argument("-m", "--model_name", default="google/switch-base-64", type=str, help="Huggingface model")
parser.add_argument("-nm", "--name_moe_layer", default="SwitchTransformersSparseMLP", type=str, help="class name of model MoELayers")
parser.add_argument("-nr", "--name_router", default="router", type=str, help="parameter name of router on MoE")
parser.add_argument("-ne", "--name_experts", default="experts", type=str, help="parameter name of router on MoE")
parser.add_argument("-nd", "--name_decoder", default="decoder", type=str, help="module name of model decoder")
parser.add_argument("-dc", "--dynamic_components", default=["wi", "wo"], type=list, help="parameter names of expert changing weights")
parser.add_argument("-sys", "--system", default="harmony", type=str)

args = parser.parse_args()

# if args.num_experts not in [8, 16, 32, 64, 128, 256]:
#     print(f"There is no model with {args.num_experts} experts")
#     exit(1)

## Validation
if args.num_iters == 0 and args.num_samples == 0:
    print("You must either specify --num_iters or --num_samples")
    exit(1)

# if args.num_iters != 0:
#     DESIRED_DATASET_SIZE = 
# else:
#     DESIRED_DATASET_SIZE = args.num_samples

def setup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    dist.init_process_group("nccl", rank=rank, world_size=args.world)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# def move_to_cuda_except_experts(model):
#     for name, module in model.named_children():
#         if name == 'experts':
#             # We want to keep the experts on cpu
#             continue
#         elif list(module.children()):
#             # If the module has children, recurse
#             move_to_cuda_except_experts(module)
#         else:
#             # If it's a leaf module (no children) and not part of experts, move to CUDA
#             module.cuda()

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        ROOT = f"{args.path}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
        setup(rank)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache")
        
        # if args.model_type == "encoder-decoder":
        #     _class = SwitchTransformersForConditionalGeneration
        # elif args.model_type == "encoder":
        #     _class = SwitchTransformersEncoderModel
        # else:
        #     print("That model type is not yet implemented!")
        #     exit(1)
        
        model = Modelling(**vars(args))
        model.cuda()
        model.eval()

        # move_to_cuda_except_experts(model)
        # model.expert_parallelise()

        flexible_dataset = FlexibleDataset(
            args.dataset, 
            tokenizer, 
            model, 
            seq_len=args.seq_len,
            num_samples=args.num_samples if args.num_samples != 0 else args.num_iters * args.batch_size * args.world
        )
        sampler = DistributedSampler(
            flexible_dataset, 
            num_replicas=args.world, 
            rank=rank, 
            shuffle=True, 
            seed=49
        )
        loader = DataLoader(
            flexible_dataset, 
            sampler=sampler, 
            batch_size=args.batch_size
        )

        if args.experiment == "standard":
            run_standard_experiment(model, tokenizer, loader, f"{ROOT}/{rank}")
            if rank == 0:
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
            model(batch)
            # outputs = model(**batch)
            # if args.model_type == "encoder-decoder":
            #     probs = F.softmax(outputs.logits, dim=-1)
            #     predicted_token_ids = torch.argmax(probs, dim=-1)
            #     predicted_token = [tokenizer.decode(pred_id, skip_special_tokens=True) for pred_id in predicted_token_ids]
            end = time.time()
            latencies.append(end-start)
    

    create_save_dir_if_not_exist(path)
    model.save_statistics(path)
    # for moe_layer in moe_layers:
    #     moe_layer.save_statistics(DIR=path)

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
        json.dump(vars(args), f)
        # json.dump({
        #     "scheduling_policy": args.schedule,
        #     "cache_policy": args.cache_policy,
        #     "batch_size": args.batch_size,
        #     "world_size": args.world,
        #     "dataset": args.dataset,
        #     "model": args.model,
        #     "seq_len": args.seq_len,
        #     "num_iters": args.num_iters,
        #     "num_samples": args.num_samples,
        #     "port": args.port,
        #     "experiment": args.experiment,
        #     "expert_cache_size": args.expert_cache_size,
        #     "eq_tokens": args.eq_tokens,
        #     "d_model": args.d_model,
        #     "name_moe_layer": args.name_moe_layer,
        #     "name_router": args.name_router,
        #     "name_experts": args.name_experts,
        #     "name_decoder": args.name_decoder,
        #     "dynamic_components": args.dynamic_components,
        # }, f)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    mp.spawn(run_inference_workload, nprocs=args.world, join=True)
    
