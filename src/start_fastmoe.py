# https://dl.acm.org/doi/pdf/10.1145/3503221.3508418
# https://github.com/laekov/fastmoe/tree/master/doc/fastermoe

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
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
from stats import Stats
from fmoe.layers import FMoE
from utils import TimedModule, get_timing_modules
from harmonymoe.router import Router, RouterConfig
from args import Args

args = Args().fastmoe()
if args.system_name != "fastmoe" and args.system_name != "fastermoe":
    raise Exception("Only fastmoe and fastermoe supported")

def setup(rank):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # FASTERMOE 
    if args.system_name == "fastermoe":
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

        model = AutoModel.from_pretrained(args.model_name, cache_dir="/cache")

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


        def create_custom_router_gate(routerConfig):
            class RouterWrapper(Router):
                def __init__(self, d_model, num_expert, world_size, top_k):
                    super().__init__(routerConfig)
                
                def forward(self, x):
                    router_mask, router_probs, router_logits = super().forward(x)
                    gate_top_k_idx = torch.argmax(router_mask, dim=-1).unsqueeze(1)
                    gate_score = router_probs

                    return gate_top_k_idx, gate_score
            
            return RouterWrapper

        def get_tensor_by_path(module, path):
            parts = path.split(".")
            current = module
            for part in parts:
                current = getattr(current, part)
            return current
        
        # Update to add FMoE to it
        def add_fmoe_model(module, idx):
            if type(module).__name__ == args.type_moe_parent:
                for child_name, child in module.named_children():
                    if type(child).__name__ == args.type_moe:
                        router = get_tensor_by_path(child, args.router_tensor_path)
                        experts = get_tensor_by_path(child, args.name_experts)
                        if isinstance(experts, nn.ModuleDict):
                            experts = list(experts.values())
                        elif isinstance(experts, nn.ModuleList):
                            experts = list(experts)

                        num_experts_per_gpu = args.num_experts // args.world_size

                        experts = experts[rank*num_experts_per_gpu:(rank+1)*num_experts_per_gpu]

                        routerConfig = RouterConfig(
                            d_model=args.d_model,
                            num_experts=args.num_experts,
                            weights=router.weight,
                            enable_skew=args.enable_router_skew,
                            enable_random=args.enable_router_random,
                            enable_uniform=args.enable_router_uniform,
                            skew=args.router_skew,
                            num_experts_skewed=args.router_num_experts_skewed,
                        )

                        new = FMoE(
                            num_expert=num_experts_per_gpu,
                            d_model=args.d_model,
                            world_size=args.world_size,
                            top_k=1,
                            expert=[lambda _, e=e: ExpertWrapper(e) for e in experts],
                            moe_group=moe_group,
                            gate=create_custom_router_gate(routerConfig)
                        )
                        
                        setattr(module, child_name, TimedModule(FMoEWrapper(new), idx=idx[0]))
                        idx[0] += 1
            else:
                for child in module.children():
                    add_fmoe_model(child, idx)

        add_fmoe_model(model, [0])

        model.cuda()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache")

        flexible_dataset = FlexibleDataset(
            args.dataset, 
            tokenizer, 
            model, 
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            model_name=args.model_name,
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

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    stats.start()

    processes = []
    mp.set_start_method('spawn', force=True)
    for i in range(args.world_size):
        p = mp.Process(target=run_inference_workload, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    stats.stop()
    stats.save(path=args.path)

    print("All done :)")