import torch
import time

# Setup your tensors
hidden_states = torch.randn(8, 29808, 768)
wis = torch.randn(8, 768, 3072)

# Benchmark bmm
start = time.time()
for _ in range(100):
    result = torch.bmm(hidden_states, wis)
print(f"bmm time: {time.time() - start}")

# Benchmark matmul
start = time.time()
for _ in range(100):
    result = torch.matmul(hidden_states, wis)
print(f"matmul time: {time.time() - start}")

# Benchmark einsum
start = time.time()
for _ in range(100):
    result = torch.einsum('bnd,bdf->bnf', hidden_states, wis)
print(f"einsum time: {time.time() - start}")