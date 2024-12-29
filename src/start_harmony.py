import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import sys
import os 
from datetime import datetime, timedelta
import time
import csv
import stat 
import json
import signal
from tqdm import tqdm
import argparse
import logging

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
from stats import Stats

from harmonymoe.utils import replace_moe_layer, get_moe_experts, get_moe_layers
from harmonymoe.moe_layer import MoEConfig, MoELayer
from router import Router

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
parser.add_argument("--num_experts", default=8, type=int)
parser.add_argument("--model_name", default="google/switch-base-64", type=str, help="Huggingface model")
parser.add_argument("--type_moe_parent", default="SwitchTransformersLayerFF", type=str, help="class name of model MoE Layer parent")
parser.add_argument("--type_moe", default="SwitchTransformersSparseMLP", type=str, help="class name of model MoE Layers")
parser.add_argument("--name_router", default="router", type=str, help="parameter name of router on MoE")
parser.add_argument("--name_experts", default="experts", type=str, help="parameter name of router on MoE")
parser.add_argument("--name_decoder", default="decoder", type=str, help="module name of model decoder")
parser.add_argument("--dynamic_components", nargs='+', default=["wi", "wo"], type=str, help="parameter names of expert changing weights")
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
parser.add_argument("--enable_router_skew", default=False, type=str2bool)
parser.add_argument("--router_skew", default=0.0, type=float, help="Value between 0 and 1")
parser.add_argument("--router_num_experts_skew", default=1, type=int, help="Number of experts that receive the skewed proportion")
parser.add_argument("--random_router_skew", default=False, type=str2bool, help="Whether to enable random skewing in the router")
parser.add_argument("--disable_async_fetch", default=False, type=str2bool, help="Whether want to disable the background expert fetching")
args = parser.parse_args()

############# LOGGING ################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
#############################################

def setup(rank, timeout=timedelta(minutes=30)):
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size, timeout=timeout)

def run_inference_workload(rank, model, batch=args.batch_size):
    try:
        logger.info(f"Starting process {rank}")
        args.batch_size = batch

        mp.current_process().name = f'Worker-{rank}'
        setup(rank)

        model.cuda()
        model.eval()

        # We need to call prepare on the MoE layers to create the cuda events
        # The cuda events cannot be created at __init__ since the model object
        # is transfered to different processes which have different GPUs
        for l in get_moe_layers(model):
            l.prepare()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name) #, cache_dir="/cache"

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
                fieldnames = ["iteration"] + list(stats[0].keys())

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
        logger.info(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        dist.destroy_process_group()

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

def signal_handler(sig, frame):
    logger.info("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         logger.info(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

if __name__ == "__main__":
    logging.info("Starting main process")

    signal.signal(signal.SIGINT, signal_handler)

    model = AutoModel.from_pretrained(args.model_name)

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
        disable_async_fetch=args.disable_async_fetch,
        model_name=args.model_name,
        num_experts=args.num_experts,
    )
    router = None 
    if args.enable_router_skew:
        router=lambda: Router(args.num_experts, skew=args.router_skew, num_expert_skew=args.router_num_experts_skew, enable_random=args.random_router_skew)
    replace_moe_layer(
        model, 
        args.type_moe_parent,
        args.type_moe, 
        args.name_router, 
        experts,
        config,
        override_router=router,
    )

    logger.info("Finished loading model")


    stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    stats.start()

    processes = []
    mp.set_start_method("spawn", force=True)
    for i in range(args.world_size):
        p = mp.Process(target=run_inference_workload, args=(i, model), kwargs=dict(batch=args.batch_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    logger.info("All processes complete")

    stats.stop()
    stats.save(path=args.path)

    logger.info("All done :)")