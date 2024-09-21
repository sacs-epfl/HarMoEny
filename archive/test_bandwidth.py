# import torch
# import time

# # def measure_bandwidth(size, source, destination, num_iters=10):
# #     tensor = torch.randn((num_iters, size), device=source)

# #     # Warmup
# #     for _ in range(5):
# #         tensor = torch.randn(size, device=source)
# #         _ = tensor.to(destination)
    
# #     torch.cuda.synchronize()

# #     start_event = torch.cuda.Event(enable_timing=True)
# #     end_event = torch.cuda.Event(enable_timing=True)

# #     start_event.record()
# #     for i in range(num_iters):
# #         _ = tensor[i].to(destination)
# #     end_event.record()

# #     end_event.synchronize()

# #     elapsed_time = start_event.elapsed_time(end_event) / 1000
# #     tensor_bytes = (tensor.nelement() * tensor.element_size()) # bytes
# #     bandwidth = (tensor_bytes  / elapsed_time) / 1e9

# #     return bandwidth

# def measure_bandwidth(size, source, destination, num_iters=10):
#     tensor_src = torch.randn(size, device=source)
#     tensor_dst = torch.zeros(size, device=destination)
    
#     # Warmup
#     for _ in range(5):
#          tensor_dst.copy_(tensor_src)
    
#     torch.cuda.synchronize()
    
#     start_time = time.perf_counter()
#     for _ in range(num_iters):
#         tensor_src += 1
#         tensor_src %= 2
#         tensor_dst.copy_(tensor_src)
#         # Perform a small computation to ensure the transfer is not optimized away
#         tensor_dst.add_(1)
#     end_time = time.perf_counter()
    
#     elapsed_time = end_time - start_time
#     tensor_bytes = tensor_src.nelement() * tensor_src.element_size()
#     bandwidth = (tensor_bytes * num_iters / elapsed_time) / 1e9  # GB/s
    
#     return bandwidth

# def main():
#     num_gpus = torch.cuda.device_count()
#     sizes = [1024, 1024*1024, 10*1024*1024, 100*1024*1024]

#     for size in sizes:
#         print(f"\nTesting with tensor size: {size} elements")

#         bandwidth = measure_bandwidth(size, "cpu", "cuda:1")
#         print(f"CPU to GPU bandwidth: {bandwidth:.2f} GB/s")

#         bandwidth = measure_bandwidth(size, "cuda:1", "cpu")
#         print(f"GPU to CPU bandwidth: {bandwidth:.2f} GB/s")

#         bandwidth = measure_bandwidth(size, "cuda:1", "cuda:2")
#         print(f"GPU to GPU bandwidth: {bandwidth:.2f} GB/s")

# if __name__ == "__main__":
#     main()

# import torch
# from torch.utils.benchmark import Timer

# def memory_bandwidth_benchmark(size, src_device, dst_device):
#     x = torch.randn(size, device=src_device)
#     y = torch.empty(size, device=dst_device)
    
#     timer = Timer(
#         stmt='y.copy_(x)',
#         globals={'x': x, 'y': y}
#     )
    
#     return timer.blocked_autorange()

# # Example usage
# result = memory_bandwidth_benchmark(int(1e8), 'cuda:0', 'cuda:1')
# print(result)

# import torch
# from torch.utils.benchmark import Timer

# def memory_bandwidth_benchmark(size, src_device, dst_device, num_runs=100):
#     x = torch.randn(size, device=src_device)
#     y = torch.empty(size, device=dst_device)
    
#     # Warmup
#     for _ in range(10):
#         y.copy_(x)
#     torch.cuda.synchronize()
    
#     timer = Timer(
#         stmt='y.copy_(x); torch.cuda.synchronize()',
#         globals={'x': x, 'y': y, 'torch': torch}
#     )
    
#     result = timer.timeit(num_runs)
    
#     # Calculate bandwidth
#     bytes_transferred = x.nelement() * x.element_size()
#     bandwidth_gb_s = (bytes_transferred / result.mean) / 1e9
    
#     return bandwidth_gb_s, result

# def run_benchmarks():
#     sizes = [1024, 1024*1024, 10*1024*1024, 100*1024*1024]
    
#     for size in sizes:
#         print(f"\nTesting with tensor size: {size} elements")
        
#         # CPU to GPU
#         bw, _ = memory_bandwidth_benchmark(size, 'cpu', 'cuda:0')
#         print(f"CPU to GPU bandwidth: {bw:.2f} GB/s")
        
#         # GPU to CPU
#         bw, _ = memory_bandwidth_benchmark(size, 'cuda:0', 'cpu')
#         print(f"GPU to CPU bandwidth: {bw:.2f} GB/s")
        
#         # GPU to GPU (if multiple GPUs are available)
#         if torch.cuda.device_count() > 1:
#             bw, _ = memory_bandwidth_benchmark(size, 'cuda:0', 'cuda:1')
#             print(f"GPU to GPU bandwidth: {bw:.2f} GB/s")
#         else:
#             print("GPU to GPU bandwidth: Not available (only one GPU detected)")

# if __name__ == "__main__":
#     run_benchmarks()

import torch
import time
import sys

def measure_bandwidth(size, source, destination, num_warmup=5, num_repeats=10):
    try:
        # Create tensors
        x = torch.rand(size, device=source)
        y = torch.empty(size, device=destination)

        # Warmup
        for _ in range(num_warmup):
            y.copy_(x)
        torch.cuda.synchronize()

        # Measure time
        start = time.perf_counter()
        for _ in range(num_repeats):
            y.copy_(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Calculate bandwidth
        elapsed_time = (end - start) / num_repeats
        bytes_transferred = x.nelement() * x.element_size()
        bandwidth_gb_s = (bytes_transferred / elapsed_time) / 1e9

        del x
        del y
        if source.startswith('cuda') or destination.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return bandwidth_gb_s
    except Exception as e:
        print(f"Error: {str(e)}...")
        exit(1)

def run_bandwidth_tests():
    # Use a large size to saturate the link, e.g., 1 GB
    #size = 2**30 // 4  # 1 GB / 4 bytes per float32
    size = 2**28 // 4

    print(f"Testing with tensor size: {size * 4 / (1024**3):.2f} GB")
    
    locations = ["cuda:0", "cuda:1", "cuda:2", 
        "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]

 
    for source in locations:
        print(f"{source}->cpu: {measure_bandwidth(size, source, 'cpu'):.2f} GB/s")
    for dest in locations:
        print(f"cpu->{dest}: {measure_bandwidth(size, 'cpu', dest):.2f} GB/s")
    
    
    # print("\t" + "\t".join(locations))
    # output = []
    # for dest in locations:
    #     if dest == source:
    #         output.append("X")
    #     else:
    #         output.append(f"{measure_bandwidth(size, source, dest):.2f}")
    # print(f"{source}\t" + "\t".join(output))

if __name__ == "__main__":
    run_bandwidth_tests()