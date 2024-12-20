import sys
from transformers import AutoModel

with open("mixtral_arch.txt", "w") as f:
    model = AutoModel.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/cache")
    sys.stdout = f
    print(model)