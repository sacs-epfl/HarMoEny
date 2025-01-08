import torch
import time

def measure_bandwidth(tensor_size, repetitions=10):
    # Ensure the tensor is pinned in memory
    cpu_tensor = torch.empty(tensor_size, dtype=torch.float32, pin_memory=True)
    gpu_tensor = torch.empty_like(cpu_tensor, device='cuda')

    # Warm-up transfers (ignored in timing)
    for _ in range(5):
        torch.cuda.synchronize()
        gpu_tensor.copy_(cpu_tensor)

    # Measure time for repeated transfers
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repetitions):
        gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate average transfer time and bandwidth
    total_time = end_time - start_time
    avg_time = total_time / repetitions
    data_transferred = cpu_tensor.numel() * cpu_tensor.element_size()  # in bytes
    bandwidth = (data_transferred / avg_time) / (1024**3)  # GB/s

    return avg_time, bandwidth

if __name__ == "__main__":
    tensor_sizes = [2**20, 2**21, 2**22, 2**23, 2**30]  # Size in number of elements
    repetitions = 100

    print(f"{'Size (MB)':<12}{'Avg Time (ms)':<15}{'Bandwidth (GB/s)':<15}")
    print("-" * 40)

    for size in tensor_sizes:
        avg_time, bandwidth = measure_bandwidth(size, repetitions)
        size_in_mb = size * 4 / (1024**2)  # Convert to MB (float32 has 4 bytes per element)
        print(f"{size_in_mb:<12.2f}{avg_time * 1e3:<15.2f}{bandwidth:<15.2f}")