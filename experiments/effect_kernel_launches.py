import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("google/switch-base-128", cache_dir="/cache")

experts = model.encoder.block[1].layer[1].mlp.experts
experts.cuda()
experts = list(experts.values())

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def execute_across_x_experts(num_experts, num_tokens):
    tokens = torch.rand((num_tokens, 768), device="cuda")
    num_tokens_per_expert = num_tokens // num_experts
    offset = 0
    start.record()
    for i in range(num_experts):
        end_offset = offset + num_tokens_per_expert
        tokens[offset:end_offset] = experts[i](tokens[offset:end_offset])
        offset = end_offset
    end.record()
    end.synchronize()
    return start.elapsed_time(end)

def perform_experiment_x_times(exp, x):
    arr = []
    for _ in range(x):
        arr.append(exp())
    return sum(arr) / len(arr)

print(perform_experiment_x_times(lambda: execute_across_x_experts(128, 12800), 10))
print(perform_experiment_x_times(lambda: execute_across_x_experts(16, 12800), 10))