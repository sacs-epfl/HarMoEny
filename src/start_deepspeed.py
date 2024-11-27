# https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf
# https://github.com/microsoft/DeepSpeed

import torch
import torch.nn as nn
import pynvml
import psutil
import torch.multiprocessing as mp 
from threading import Thread
import sys
import os 
import time
import csv
import json
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
import argparse 

from utils import TimedModule, get_timing_modules
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersBlock, SwitchTransformersLayerSelfAttention, SwitchTransformersLayerCrossAttention

import deepspeed

parser = argparse.ArgumentParser(
    prog="Run inference on FastMoE",
)
parser.add_argument("--dataset", default="sst2", type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("--seq_len", default=120, type=int)
parser.add_argument("--path", default="outputs/out", type=str, help="Specify where to save path")
parser.add_argument("--num_experts", default=8, type=int, help="Number of experts we want to match dense model to")
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("--local_rank", default=0, type=int) 
parser.add_argument("--world_size", default=8, type=int)
args = parser.parse_args()


############# GLOBAL AFFAIRS ################
pynvml.nvmlInit()
#############################################

def setup():
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)

def run_inference_workload():
    setup()

    model_name = f"google/switch-base-{args.num_experts}"
    model = AutoModel.from_pretrained(model_name, cache_dir="/cache")

    class MLPWrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child
        
        def forward(self, x):
            x = self.child(x)
            return x[0], (x[1], x[2])

    # Update to add DeepspeedMoE to it
    def add_deepspeed_moe_model(module, idx):
        if type(module).__name__ == "SwitchTransformersLayerFF":
            for child_name, child in module.named_children():
                if type(child).__name__ == "SwitchTransformersSparseMLP":
                    router = getattr(child, "router")
                    experts = getattr(child, "experts")
                    if type(experts) == nn.ModuleDict:
                        experts = list(experts.values())
                    
                    num_experts_per_gpu = args.num_experts // args.world_size

                    experts = experts[args.local_rank*num_experts_per_gpu:(args.local_rank+1)*num_experts_per_gpu]

                    new = deepspeed.moe.layer.MoE(
                        hidden_size=768,
                        expert=experts[0],
                        num_experts=args.num_experts,
                        ep_size=args.world_size,
                        k=1,
                        eval_capacity_factor=10.0,
                        #drop_tokens=False,
                        use_tutel=True,
                        top2_2nd_expert_sampling=False,
                        use_rts=False,
                    )

                    with torch.no_grad():
                        new.deepspeed_moe.gate.wg.weight.copy_(router.classifier.weight)
                        for i in range(len(experts)):
                            new.deepspeed_moe.experts.deepspeed_experts[i].wi.weight.copy_(experts[i].wi.weight)
                            new.deepspeed_moe.experts.deepspeed_experts[i].wo.weight.copy_(experts[i].wo.weight)

                    setattr(module, child_name, TimedModule(MLPWrapper(new), idx=idx[0]))
                    idx[0] += 1
        else:
            for child in module.children():
                add_deepspeed_moe_model(child, idx)

    add_deepspeed_moe_model(model, [0])

    model.eval()
    model.cuda()

    ds_engine = deepspeed.init_inference(
        model,
        dtype=torch.float,
        replace_with_kernel_inject=False,
        moe={
            "enabled": True,
            "ep_size": args.world_size, 
            "moe_experts": [args.num_experts],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cache")

    flexible_dataset = FlexibleDataset(
        args.dataset, 
        tokenizer, 
        model, 
        seq_len=args.seq_len,
        num_samples=args.num_samples
    )
    sampler = DistributedSampler(
        flexible_dataset, 
        num_replicas=args.world_size, 
        rank=args.local_rank, 
        shuffle=True, 
        seed=49
    )
    loader = DataLoader(
        flexible_dataset, 
        sampler=sampler, 
        batch_size=args.batch_size
    )

    latencies, run_start, run_end = run_standard_experiment(ds_engine, loader)

    path = f"{args.path}/{args.local_rank}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    ############# E2E #######################
    file_path = f"{path}/e2e.csv"
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
        file_path = f"{path}/layer-{timing_module.idx}.csv"
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
        json.dump({ "start": run_start, "end": run_end, **run_info, "world_size": args.world_size}, f, indent=4)

def run_standard_experiment(ds_engine, loader):
    latencies = []
    
    with torch.no_grad():
        # WARMUP
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            ds_engine(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )
            itr += 1
            if itr == args.warmup_rounds:
                break

        # RUN ACTUAL EXPERIMENT
        run_start = time.time()
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
        run_end = time.time()
    
    return latencies, run_start, run_end

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

if __name__ == "__main__":
    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics))
    metric_thread.start()

    run_inference_workload()

    stop_event.set()
    metric_thread.join()

    df = pd.DataFrame(metrics)
    df.to_csv(f"{args.path}/stats.csv")

    print("All done :)")

    pynvml.nvmlShutdown()