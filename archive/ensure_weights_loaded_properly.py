import torch

original = torch.load("original.pth", map_location=torch.device("cpu"))
modified = torch.load("modified.pth", map_location=torch.device("cpu"))

count_mismatch = 0
for name, weights in original.items():
    if "router.classifier" in name:
        parts = name.split(".")
        name = f"{'.'.join(parts[:6])}.deepspeed_moe.gate.wg.weight"
    elif "experts.expert_" in name:
        parts = name.split(".")
        name = f"{'.'.join(parts[:6])}.deepspeed_moe.experts.deepspeed_experts.{parts[7].split('_')[1]}.{parts[8]}.weight"
    
    if not torch.equal(weights, modified[name]):
            print(f"{name} is not the sae across the modification")
            count_mismatch += 1

print(f"THE AMOUNT OF MODULES THAT HAVE DIFFERENT WEIGHTS: {count_mismatch}")