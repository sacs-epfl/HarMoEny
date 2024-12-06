import torch
from transformers import AutoModel
import copy

torch.cuda.set_device(0)

model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")
expert1 = model.encoder.block[1].layer[1].mlp.experts.expert_0
expert2 = model.encoder.block[1].layer[1].mlp.experts.expert_1
gpu_expert = copy.deepcopy(expert1).cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Time to swap
def swap():
    expert1.cuda()
    expert2.cpu()
    torch.cuda.empty_cache()

    start.record()
    expert1.cpu()
    torch.cuda.synchronize()
    expert2.cuda()
    torch.cuda.synchronize()
    end.record()
    end.synchronize()

    return start.elapsed_time(end)

def overwrite():
    torch.cuda.empty_cache()

    start.record()
    with torch.no_grad():
        gpu_expert.wi.weight.copy_(expert1.wi.weight)
        gpu_expert.wo.weight.copy_(expert1.wo.weight)
    torch.cuda.synchronize()
    end.record()
    end.synchronize()

    return start.elapsed_time(end)

def perform_experiment_x_times(exp, x):
    arr = []
    for _ in range(x):
        arr.append(exp())
    return sum(arr) / len(arr)

print(perform_experiment_x_times(swap, 10))
print(perform_experiment_x_times(overwrite, 10))
