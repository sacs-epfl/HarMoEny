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
from args import Args
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
from stats import Stats

from harmonymoe.utils import replace_moe_layer, get_moe_experts, get_moe_layers
from harmonymoe.moe_layer import MoEConfig, MoELayer
from router import Router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
args = Args().harmony()

# rank = int(os.environ.get("RANK", 0))
# world_size = int(os.environ.get("WORLD_SIZE", 1))

def setup(rank, timeout=timedelta(minutes=30)):
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    torch.set_num_threads(os.cpu_count() // args.world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # dist.init_process_group(
    #     "nccl", rank=rank, world_size=world_size, timeout=timeout
    # )
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=rank,
        world_size=args.world_size,
        timeout=timeout
    )

def generate_model(rank):
    model = AutoModel.from_pretrained(args.model_name)

    # experts = get_moe_experts(model, args.type_moe, args.name_experts)

    # for layer_experts in experts:
    #     for expert in layer_experts:
    #         for param in expert.parameters():
    #             param.data.share_memory_()

    # for layer_experts in experts:
    #     for expert in layer_experts:
    #         for param in expert.parameters():
    #             param.data = param.data.pin_memory()

    # pinned_experts = []
    # for layer_experts in experts:
    #     layer_pinned = []
    #     for expert in layer_experts:
    #         # Create a pinned state_dict for each expert
    #         pinned_state_dict = {
    #             name: param.detach().cpu().pin_memory()
    #             for name, param in expert.state_dict().items()
    #         }
    #         layer_pinned.append(pinned_state_dict)
    #     pinned_experts.append(layer_pinned)

    config = MoEConfig(
        rank=rank,
        world_size=args.world_size,
        scheduling_policy=args.scheduling_policy,
        cache_policy=args.cache_policy,
        expert_cache_size=args.expert_cache_size,
        eq_tokens=args.eq_tokens,
        d_model=args.d_model,
        expert_placement=args.expert_placement,
        fetching_strategy=args.expert_fetching_strategy,
        model_name=args.model_name,
        num_experts=args.num_experts,
        enable_skew=args.enable_router_skew,
        enable_random=args.enable_router_random,
        enable_uniform=args.enable_router_uniform,
        skew=args.router_skew,
        num_experts_skewed=args.router_num_experts_skewed,
    )

    replace_moe_layer(
        model,
        args.type_moe_parent,
        args.type_moe,
        args.name_experts,
        args.router_tensor_path,
        config,
    )

    logger.info("Finished loading model")

    return model

#def run_inference_workload(rank, model):
def run_inference_workload(rank):
    try:
        logger.info(f"Starting process {rank}")

        setup(rank)
        
        mp.current_process().name = f"Worker-{rank}"

        model = generate_model(rank)
        model.cuda()
        model.eval()

        # # We need to call prepare on the MoE layers to create the cuda events
        # # The cuda events cannot be created at __init__ since the model object
        # # is transfered to different processes which have different GPUs
        # for l in get_moe_layers(model):
        #     l.prepare()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
            num_replicas=args.world_size,
            rank=rank,
            shuffle=True,
            seed=49,
        )

        loader = DataLoader(
            flexible_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
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
                writer.writerow({"iteration": idx, "latency (s)": latency})

        for l in get_moe_layers(model):
            file_path = f"{path}/moe_layer-{l.layer_idx}.csv"
            stats = l.get_statistics()[args.warmup_rounds :]

            with open(file_path, "w") as f:
                fieldnames = ["iteration"] + list(stats[0].keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(len(stats)):
                    dic = {"iteration": i, **stats[i]}

                    writer.writerow(dic)

        if rank == 0:
            run_info = vars(args).copy()
            with open(f"{args.path}/data.json", "w") as f:
                json.dump({"start": run_start, "end": run_end, **run_info}, f, indent=4)

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
            latencies.append(end - start)
        run_end = time.time()

    return latencies, run_start, run_end


# def signal_handler(sig, frame):
#     logger.info("Main process received Ctrl+C! Terminating all child processes...")
#     for child in mp.active_children():
#         logger.info(f"Terminating child process PID: {child.pid}")
#         child.terminate()
#     sys.exit(0)



#def launch_processes_and_wait(model):
def launch_processes_and_wait():
    processes = [None] * args.world_size

    def start_process(i):
        process = mp.Process(
            target=run_inference_workload,
            args=(i,),
        )
        process.start()
        return i, process
    
    mp.set_start_method("spawn", force=True)

    with ThreadPoolExecutor(max_workers=args.world_size) as executor:
        futures = [executor.submit(start_process, i) for i in range(args.world_size)]
        for fut in as_completed(futures):
            idx, process = fut.result()
            processes[idx] = process
    
    for process in processes:
        process.join()

    logger.info("All processes complete")

if __name__ == "__main__":
    logging.info(f"Starting Main Process")

    # signal.signal(signal.SIGINT, signal_handler)

    #model = generate_model()

    # stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    # stats.start()

    #launch_processes_and_wait(model)
    launch_processes_and_wait()

    # stats.stop()
    # stats.save(path=args.path)

    logger.info("All done :)")
