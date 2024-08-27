from transformers import SwitchTransformersEncoderModel, AutoTokenizer
from datasets import load_dataset
import nvidia_smi
import torch
import time


# FOR MODEL
print("MODEL")
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

pre = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8")
model.to("cuda:0")
torch.cuda.synchronize()
post = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
print(f"Static memory footprint of full model: {(post-pre) / 10**6} MB")

# Reset
model.to("cpu")
torch.cuda.empty_cache()
torch.cuda.synchronize()
pre = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
model.encoder.block[1].layer[1].mlp.experts.expert_0.to("cuda:0")
torch.cuda.synchronize()
post = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
print(f"Static memory footprint of a single expert: {(post-pre) / 10**6} MB")


# FOR DATASET
print("DATASET")
dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=False, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

total_tokens = 0
num_rows = 0

try:
    for entry in dataset:
        tokens = tokenizer.encode(entry["text"])
        total_tokens += len(tokens)
        num_rows += 1
except KeyboardInterrupt:
        print("\nProcess interrupted by user. Returning current counts.")


print(f"Total number of tokens: {total_tokens}")
print(f"Total number of rows: {num_rows}")
print(f"Avg number of tokens per row: {total_tokens/num_rows}")

