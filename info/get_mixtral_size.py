from transformers import AutoModel

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  
model = AutoModel.from_pretrained(model_name, cache_dir="../../cache")

total_params = sum(p.numel() for p in model.parameters())
size_in_gb = total_params * 4 / (1024 ** 3)  # Convert bytes to GB

print(f"Model size (approximate): {size_in_gb:.2f} GB")

