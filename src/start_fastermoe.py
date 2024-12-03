# https://dl.acm.org/doi/pdf/10.1145/3503221.3508418
# https://github.com/laekov/fastmoe/tree/master/doc/fastermoe

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import pynvml
import psutil
from threading import Thread
import sys
import os 
import time
import csv
import json
import pandas as pd
import signal
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
import argparse 

from fmoe.gates.naive_gate import NaiveGate
from fmoe.layers import FMoE
from router import Router 
from utils import TimedModule, get_timing_modules

parser = argparse.ArgumentParser(
    prog="Run inference on FastMoE",
)
parser.add_argument("--dataset", default="sst2", type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("--seq_len", default=120, type=int)
parser.add_argument("--path", default="outputs/out", type=str, help="Specify where to save path")
parser.add_argument("--num_experts", default=8, type=int, help="Number of experts we want to match dense model to")
parser.add_argument("--world_size", default=torch.cuda.device_count(), type=int, help="Number of GPUs to use")
parser.add_argument("--port", default="1234", type=str)
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("--router_skew", default=0.0, type=float, help="Value between 0 and 1")
args = parser.parse_args()


############# GLOBAL AFFAIRS ################
pynvml.nvmlInit()
#############################################

def setup(rank):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # FASTERMOE 
    # https://github.com/laekov/fastmoe/tree/master/doc/fastermoe
    ## Smart Scheduling
    os.environ["FMOE_FASTER_SCHEDULE_ENABLE"] = "1"
    os.environ["FMOE_FASTER_GROUP_SIZE"] = "2"

    ## Expert Shadowing
    os.environ["FMOE_FASTER_SHADOW_ENABLE"] = "1"
    os.environ["FMOE_FASTER_GLBPLC_NETBW"] = "10.13e12" # Run the get_gemm in info
    os.environ["FMOE_FASTER_GLBPLC_GPUTP"] = "7.19" # This is the slowest where two GPUs are not directly connected via NVLink
    os.environ["FMOE_FASTER_GLBPLC_ALPHA"] = "4" # 3072 / 768
    os.environ["FMOE_FASTER_GLBPLC_DMODEL"] = "768"

    torch.cuda.set_device(rank)

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        setup(rank)

        moe_group = dist.new_group(ranks=list(range(args.world_size)))

        model_name = f"google/switch-base-{args.num_experts}"
        model = AutoModel.from_pretrained(model_name, cache_dir="/cache")

        class ExpertWrapper(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.child = child
            
            def forward(self, x, _):
                return self.child(x)
        
        class FMoEWrapper(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.child = child

            def forward(self, x):
                batch_size, seq_len, d_model = x.shape
                x = x.reshape(-1, d_model)  # Shape: [batch_size * seq_len, d_model]
                output = self.child(x)  # FMoE expects input of shape [tokens, d_model]
                output = output.reshape(batch_size, seq_len, d_model)
                return output
        
        class RouterWrapper(Router):
            def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
                super().__init__(args.num_experts, skew=args.router_skew)
                self.d_model = d_model
            
            def forward(self, x):
                router_mask, router_probs, router_logits = super().forward(x)
                gate_top_k_idx = torch.argmax(router_mask, dim=-1).unsqueeze(1)
                gate_score = router_probs

                return gate_top_k_idx, gate_score

        # Update to add FMoE to it
        def add_fmoe_model(module, idx):
            if type(module).__name__ == "SwitchTransformersLayerFF":
                for child_name, child in module.named_children():
                    if type(child).__name__ == "SwitchTransformersSparseMLP":
                        router = getattr(child, "router")
                        experts = getattr(child, "experts")
                        if type(experts) == nn.ModuleDict:
                            experts = list(experts.values())
                        
                        num_experts_per_gpu = args.num_experts // args.world_size

                        experts = experts[rank*num_experts_per_gpu:(rank+1)*num_experts_per_gpu]

                        new = FMoE(
                            num_expert=num_experts_per_gpu,
                            d_model=768,
                            world_size=args.world_size,
                            top_k=1,
                            expert=[lambda _, e=e: ExpertWrapper(e) for e in experts],
                            moe_group=moe_group,
                            gate=RouterWrapper,
                        )

                        # with torch.no_grad():
                        #     new.gate.gate.weight.copy_(router.classifier.weight)
                        
                        setattr(module, child_name, TimedModule(FMoEWrapper(new), idx=idx[0]))
                        idx[0] += 1
            else:
                for child in module.children():
                    add_fmoe_model(child, idx)

        add_fmoe_model(model, [0])

        model.cuda()
        model.eval()

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

        path = f"{args.path}/{rank}"
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

    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics))
    metric_thread.start()

    processes = []
    mp.set_start_method('spawn', force=True)
    for i in range(args.world_size):
        p = mp.Process(target=run_inference_workload, args=(i,))
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