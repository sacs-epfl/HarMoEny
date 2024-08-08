# # For this script I am going to do an encoder model only MoE which will allow us to test the speed of doing a naive 
# # expert parallelism. The Encoder phase would be more interesting as it involves a lot of tokens rather than the 
# # decoder which does only one token at a time.

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SwitchTransformersEncoderModel
# from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
# import torch
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# import time
# import pandas as pd
# import csv

# import os
# os.environ["HF_HOME"] = "/cache"
# os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

# FIXED_LENGTH = 30
# NUM_ITERS = 100
# BATCHES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
# # BATCHES_TO_TEST = [4, 8]
# NUM_TOKENS_TO_TEST = [120]

# tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
# model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8", num_gpus=4)
# # def print_model_device(model):
# #     for name, param in model.named_parameters():
# #         print(f"Parameter: {name} is on device: {param.device}")

# # # Example usage
# # print_model_device(model)
# # print(model)
# # exit(1)

# dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)


# # Instead of creating batches I want to append things together to get the correct length
# def create_batches(dataset, batch_size):
#     batches = []
#     index = 0

#     for batch_samples in DataLoader(dataset, batch_size=batch_size, shuffle=False):
#         tokens = tokenizer(batch_samples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=FIXED_LENGTH)
#         batches.append(tokens)
#         if len(batches) >= NUM_ITERS:
#             break

#     return batches

# # def create_batches(dataset, token_length):
# #     batches = []
# #     cur = {
# #         "input_ids": [],
# #         "attention_mask": [],
# #     }
# #     for entry in dataset:
# #         tokens = tokenizer(entry["text"], return_tensors="pt", max_length=token_length, truncation=True)
# #         cur["input_ids"].extend(tokens["input_ids"][0].tolist())
# #         cur["attention_mask"].extend(tokens["attention_mask"][0].tolist())
# #         if len(cur["input_ids"]) > token_length:
# #             cur["input_ids"] = torch.LongTensor([cur["input_ids"][:token_length]])
# #             cur["attention_mask"] = torch.LongTensor([cur["attention_mask"][:token_length]])
# #             batches.append(cur)
# #             if len(batches) >= NUM_ITERS:
# #                 break
# #             cur = {
# #                 "input_ids": [],
# #                 "attention_mask": [],
# #             }
# #     return batches


# def run_experiment(batch_size):
#     # subset = list(dataset.take(batch_size * NUM_ITERS))
#     batches = create_batches(dataset, batch_size)
#     latencies = []
#     for batch in batches:
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             output = model(**batch)
#         torch.cuda.synchronize()
#         end = time.time()
#         latencies.append(end-start)
#     return latencies

# output_data = {}
# for batch_size in BATCHES_TO_TEST:
#     output_data[batch_size*FIXED_LENGTH] = run_experiment(batch_size)
# # for token_length in NUM_TOKENS_TO_TEST:
# #     output_data[token_length] = run_experiment(token_length)

# DIR = "outputs/latency_measurements_naive_expert_parallel_switch_8"
# if not os.path.exists(DIR):
#     os.mkdir(DIR)

# with open(f"{DIR}/e2e.csv", "w") as f:
#     fieldnames = ["Number of Tokens", "Iteration Number", "Latency (s)"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     for num_tokens, latencies in output_data.items():
#         for idx, latency in enumerate(latencies):
#             writer.writerow({"Number of Tokens": num_tokens, "Iteration Number": idx, "Latency (s)": latency})

# model.expert_save_latencies(DIR)

# print("Latency measurements saved!")


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

import os
os.environ["HF_HOME"] = "/cache"
os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

FIXED_LENGTH = 30
NUM_ITERS = 10
# BATCHES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCHES_TO_TEST = [8192]

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
        print(end-start)
        latencies.append(end-start)
    return latencies

output_data = {}
torch.cuda.cudart().cudaProfilerStart()
for batch_size in BATCHES_TO_TEST:
    torch.cuda.nvtx.range_push(f"BATCH SIZE EXPERIMENT: {batch_size}")
    output_data[batch_size*FIXED_LENGTH] = run_experiment(batch_size)
    torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()


DIR = "outputs/latency_measurements_naive_expert_parallel_switch_8_test"
if not os.path.exists(DIR):
    os.mkdir(DIR)

with open(f"{DIR}/e2e.csv", "w") as f:
    fieldnames = ["Number of Tokens", "Iteration Number", "Latency (s)"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for num_tokens, latencies in output_data.items():
        for idx, latency in enumerate(latencies):
            writer.writerow({"Number of Tokens": num_tokens, "Iteration Number": idx, "Latency (s)": latency})

model.expert_save_latencies(DIR)

print("Latency measurements saved!")
