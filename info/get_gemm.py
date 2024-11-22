import torch
import time

# Define matrix sizes (e.g., 4096 x 4096 x 4096 for large matrices)
M, K, N = 4096, 4096, 4096

# Allocate matrices on GPU
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')
C = torch.empty(M, N, device='cuda')

# Measure GeMM performance
start = time.time()
for _ in range(10):  # Repeat to get average time
    torch.matmul(A, B, out=C)
torch.cuda.synchronize()  # Ensure all operations are done
end = time.time()

# Calculate GFLOPS
num_operations = 2 * M * K * N
elapsed_time = (end - start) / 10
gflops = num_operations / elapsed_time / 1e12
print(f"Time: {elapsed_time:.4f} s, Performance: {gflops:.2f} TFLOPS")