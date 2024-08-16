from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SwitchTransformersEncoderModel
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
import pandas as pd
import csv
import torch.nn as nn
from functools import wraps
import nvtx
from datetime import datetime

import torch.multiprocessing as mp

import os
os.environ["HF_HOME"] = "/cache"
os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

import sys
if len(sys.argv) < 2:
    print("Please provide filename as first argument")
    exit(1)

torch.set_num_threads(12)

FIXED_LENGTH = 30
NUM_ITERS = 25 #25
# BATCHES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCHES_TO_TEST = [8192]

DIR = f"outputs/latency_measurements_{sys.argv[1]}_expert_parallel_switch_8/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8", num_gpus=4)
    model.encoder.expert_parallelise()

    # Add nvtx to the model
    def nvtx_annotate(message, color="blue"):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with nvtx.annotate(message, color=color):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def annotate_model_forward(model, name_prefix=""):
        for name, module in model.named_modules():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(module, nn.Module):
                if hasattr(module, "forward"):
                    # Wrap the forward method with the NVTX decorator
                    module.forward = nvtx_annotate(f"{full_name}.forward", color="green")(module.forward)
        print(f"Annotated the model")

    annotate_model_forward(model)

    dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)

    # Instead of creating batches I want to append things together to get the correct length
    def create_batches(dataset, batch_size):
        batches = []
        index = 0

        for batch_samples in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            tokens = tokenizer(batch_samples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=FIXED_LENGTH)
            batches.append(tokens)
            if len(batches) >= NUM_ITERS:
                break

        return batches

    def run_experiment(batch_size):
        batches = create_batches(dataset, batch_size)
        latencies = []
        for idx, batch in enumerate(batches):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                torch.cuda.nvtx.range_push(f"BATCH {idx}")
                output = model(**batch)
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end-start)
        return latencies

    output_data = {}
    torch.cuda.cudart().cudaProfilerStart()

    start = time.time()
    for batch_size in BATCHES_TO_TEST:
        torch.cuda.nvtx.range_push(f"BATCH SIZE EXPERIMENT: {batch_size}")
        output_data[batch_size*FIXED_LENGTH] = run_experiment(batch_size)
        torch.cuda.nvtx.range_pop()
    end = time.time()

    with open(f"{DIR}/e2e.csv", "w") as f:
        fieldnames = ["Number of Tokens", "Iteration Number", "Latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for num_tokens, latencies in output_data.items():
            for idx, latency in enumerate(latencies):
                writer.writerow({"Number of Tokens": num_tokens, "Iteration Number": idx, "Latency (s)": latency})

    with open(f"{DIR}/run_time.txt", "w") as f:
        f.write(str(end-start))
        f.close()

    js = {
        "scheduler_name": sys.argv[1]
    }
    with open(f"{DIR}/info.json", "w") as f:
        json.dump(js, f)

    model.expert_save_latencies(DIR)

    print("Latency measurements saved!")

    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

