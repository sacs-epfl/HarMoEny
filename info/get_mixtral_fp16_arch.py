import sys
from transformers import AutoModel
import torch

with open("mixtral_fp16_arch.txt", "w") as f:
    model = AutoModel.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.float16)
    print(model.layers[0].block_sparse_moe.experts[0].w1.weight.dtype)
    sys.stdout = f
    print(model)