# For this script I am going to do an encoder model only MoE which will allow us to test the speed of doing a naive 
# expert parallelism. The Encoder phase would be more interesting as it involves a lot of tokens rather than the 
# decoder which does only one token at a time.

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, SwitchTransformersEncoderModel
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP
import torch
from datasets import load_dataset
import time
import pandas as pd
import csv

import os
os.environ["HF_HOME"] = "/cache"
os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

FIXED_LENGTH = 30
NUM_ITERS = 100
# BATCHES_TO_TEST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCHES_TO_TEST = [8]

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8", num_gpus=4)

dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)

def create_batches(subset, batch_size):
    batches = []
    for i in range(0, len(subset), batch_size):
        batch_samples = subset[i:i+batch_size]
        texts = [sample["text"] for sample in batch_samples]
        tokens = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=FIXED_LENGTH)
        batches.append(tokens)
    return batches

def run_experiment(batch_size):
    subset = list(dataset.take(batch_size * NUM_ITERS))
    batches = create_batches(subset, batch_size)
    latencies = []
    for batch in batches:
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(**batch)
        torch.cuda.synchronize()
        end = time.time()
        latencies.append(end-start)
    return latencies

output_data = {}
for batch_size in BATCHES_TO_TEST:
    output_data[batch_size*FIXED_LENGTH] = run_experiment(batch_size)

DIR = "outputs/latency_measurements_naive_expert_parallel_switch_8"
if not os.path.exists(DIR):
    os.mkdir(DIR)

with open(f"{DIR}/e2e.csv", "w") as f:
    fieldnames = ["Number of Tokens", "Iteration Number", "Latency (s)"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for num_tokens, latencies in output_data.items():
        for idx, latency in enumerate(latencies):
            writer.writerow({"Number of Tokens": num_tokens, "Iteration Number": idx, "Latency (s)": latency})

# df = pd.DataFrame(columns=columns)
# for total_tokens, latencies in output_data.items():
#     for idx, latency in enumerate(latencies):
#         df_row = pd.DataFrame([[total_tokens, idx, latency]], columns=columns)
#         df = pd.concat([df, df_row], ignore_index=True)
# df.to_csv(f"{DIR}/e2e.csv", index=False)

model.expert_save_latencies(DIR)

print("Latency measurements saved!")

