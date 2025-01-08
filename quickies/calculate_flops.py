import torch
import time

def measure_flops(a, b, c, repetitions=10):
    # Generate random matrices on GPU
    mat1 = torch.randn(a, b, device='cuda')
    mat2 = torch.randn(b, c, device='cuda')

    # Warm-up (ignored in timing)
    for _ in range(5):
        torch.cuda.synchronize()
        torch.matmul(mat1, mat2)

    # Measure time for repeated multiplications
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repetitions):
        result = torch.matmul(mat1, mat2)
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate average execution time and FLOPs
    total_time = end_time - start_time
    avg_time = total_time / repetitions
    #flops = (2 * a * b * c - a * c) / avg_time  # FLOPs per second
    flops = (a * c * (2 * b - 1)) / avg_time
    gflops = flops / 1e9  # Convert to GFLOPs

    return avg_time, gflops

if __name__ == "__main__":
    # Define matrix sizes
    a, b, c = 4096, 4096, 4096  # Example: square matrices
    repetitions = 100

    avg_time, gflops = measure_flops(a, b, c, repetitions)
    print(f"Matrix size: {a}x{b} * {b}x{c}")
    print(f"Average time per multiplication: {avg_time * 1e3:.2f} ms")
    print(f"Performance: {gflops:.2f} GFLOPs")
