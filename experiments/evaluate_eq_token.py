import sys

from transformers import AutoModel
import torch


AVG_OVER_NUM_ITERS = 5

if len(sys.argv) < 2:
    print("Please provide as argument the eq token number")
    exit(1)

eq = int(sys.argv[1])

model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")
expert_0 = model.encoder.block[1].layer[1].mlp.experts.expert_0.to("cuda:0")
expert_1 = model.encoder.block[1].layer[1].mlp.experts.expert_1.to("cuda:0")

dim = model.config.d_model

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def create_num_tokens_tensor(d):
    return torch.rand((d, dim), device="cuda:0")

def get_average_time_for_num_tokens(num_tokens_expert_0, num_tokens_expert_1=0):
    # Warmup
    expert_0(create_num_tokens_tensor(num_tokens_expert_0))
    if num_tokens_expert_1 != 0:
        expert_1(create_num_tokens_tensor(num_tokens_expert_1))

    _sum = 0
    for _ in range(AVG_OVER_NUM_ITERS):
        tokens_0 = create_num_tokens_tensor(num_tokens_expert_0)
        if num_tokens_expert_1 != 0:
            tokens_1 = create_num_tokens_tensor(num_tokens_expert_1)
        torch.cuda.synchronize()
        start.record()
        expert_0(tokens_0)
        if num_tokens_expert_1 != 0:
            expert_1(tokens_1)
        end.record()
        torch.cuda.synchronize()
        _sum += start.elapsed_time(end)
    
    return _sum / AVG_OVER_NUM_ITERS

def test_one():
    result = get_average_time_for_num_tokens(50, 50)
    result_eq = get_average_time_for_num_tokens(100+eq)

    print(f"{result} vs {result_eq}: diff {result-result_eq}")

def test_two():
    result = get_average_time_for_num_tokens(100, 50)
    result_eq = get_average_time_for_num_tokens(150+eq)

    print(f"{result} vs {result_eq}: diff {result-result_eq}")

def test_three():
    result = get_average_time_for_num_tokens(75, 75)
    result_eq = get_average_time_for_num_tokens(150+eq)

    print(f"{result} vs {result_eq}: diff {result-result_eq}")

test_one()
test_two()
test_three()



