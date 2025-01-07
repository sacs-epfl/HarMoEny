import sys
from transformers import AutoModel

with open("qwen_arch.txt", "w") as f:
    model = AutoModel.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat", cache_dir="/cache")
    sys.stdout = f
    print(model)