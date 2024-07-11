import torch
import csv

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

EXPERIMENTATION_NUM_ITERS = 100

def experiment(_from, _to):
    latencies = []
    # Warmup
    tensor = torch.rand((768, 3072), dtype=torch.float32)
    tensor.to(_from)
    tensor.to(_to)

    # Experiment
    for i in range(EXPERIMENTATION_NUM_ITERS):
        tensor = torch.rand((768, 3072), dtype=torch.float32)
        tensor.to(_from)
        start.record()
        tensor.to(_to)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        latencies.append(elapsed_time)
    return latencies


options = ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]


with open("outputs/latencies_moving_9437184_byte_tensor.csv", "w") as f:
    fieldnames = ["x", "y", "latency (ms)"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for x in options:
        for y in options:
            res = experiment(x, y)
            writer.writerow({"x": x, "y": y, "avg latency (ms)": res})