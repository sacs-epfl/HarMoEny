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
from copy import deepcopy
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from bitsandbytes.functional import dequantize_4bit

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from awq import AutoAWQForCausalLM

from flexible_dataset import FlexibleDataset
from stats import Stats

from harmonymoe.utils import replace_moe_layer, get_moe_experts, get_moe_layers
from harmonymoe.moe_layer import MoEConfig, MoELayer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
args = Args().harmony()

def setup(rank, timeout=timedelta(minutes=30)):
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    torch.set_num_threads(os.cpu_count() // args.world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        rank=rank,
        world_size=args.world_size,
        timeout=timeout
    )

def generate_model(rank):
    if args.model_dtype == "int8" and args.loader == "transformers":
        #if rank == 0:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float32,
        )

        model = AutoModel.from_pretrained(
            args.model_name, 
            quantization_config=quantization_config, 
        )

        # print(model.config)
        # exit(0)

        model.cpu()
        torch.cuda.empty_cache()  # Run after moving to CPU

        device = f"cuda:{rank}"
        if "Mixtral" in args.model_name:
            for layer in model.layers:
                moe = layer.block_sparse_moe
                for expert_idx in range(len(moe.experts)):
                    original_expert = moe.experts[expert_idx]
                                        
                    # Convert each linear layer
                    for name, module in original_expert.named_children():
                        if isinstance(module, Linear4bit):
                            module = module.to(device)

                            quantized_data = module.weight.data  # Typically torch.uint8
                            quant_state = module.quant_state  # Contains scale and zero point
                            
                            # Dequantize the 4-bit weights
                            decoded_weights = dequantize_4bit(quantized_data, quant_state)
                            decoded_weights = decoded_weights.view(module.out_features, module.in_features)
                            decoded_weights = decoded_weights.to("cpu")
                            module = module.to("cpu")

                            new_linear = torch.nn.Linear(
                                module.in_features,
                                module.out_features,
                                bias=module.bias is not None,
                                dtype=torch.float32,
                            )

                            with torch.no_grad():
                                new_linear.weight.copy_(decoded_weights) #.half()

                                if module.bias is not None:
                                    new_linear.bias.copy_(module.bias.data.float()) #.half()

                            setattr(original_expert, name, new_linear)

        #     model.save_pretrained("/cache/quant")
        #     print("Finished saving model. If you would like to use it set path and run without special dtype")
        # return None

    elif args.model_dtype == "int4" and args.loader == "transformers":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float32,
        )

        model = AutoModel.from_pretrained(
            args.model_name, 
            quantization_config=quantization_config, 
        )

        model.cpu()
        torch.cuda.empty_cache()  # Run after moving to CPU
    else:
        dtype = "auto"
        if args.model_dtype == "float16":
            dtype = torch.float16
        elif args.model_dtype == "float":
            dtype = torch.float

        if args.loader == "transformers":
            model = AutoModel.from_pretrained(args.model_name, torch_dtype=dtype)
        elif args.loader == "awq":
            model = AutoAWQForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            raise ValueError(f"{args.loader} is not a valid loader")

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

def run_inference_workload(rank):
    try:
        logger.info(f"Starting process {rank}")

        setup(rank)
        
        mp.current_process().name = f"Worker-{rank}"

        model = generate_model(rank)
        if model is None:
            return

        model.to(f"cuda:{rank}")
        model.eval()

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
            run_info["system_name"] = "exflow" if args.scheduling_policy == "exflow" else "harmony"
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

def signal_handler(sig, frame):
    logger.info("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
        logger.info(f"Terminating child process PID: {child.pid}")
        child.terminate()
    sys.exit(0)

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

    signal.signal(signal.SIGINT, signal_handler)

    stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    stats.start()

    launch_processes_and_wait()

    stats.stop()
    stats.save(path=args.path)

    logger.info("All done :)")
