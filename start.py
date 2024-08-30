import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import sys

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size, port="12345"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_inference_workload(rank, world_size, port, scheduling_policy):
    if rank == 0:
        DIR = f"outputs/latency_measurements_{scheduling_policy}/{port}/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"

    setup(rank, world_size, port)

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersEncoderModel.from_pretrained("google/switch-base-8")
    model.cuda()

    dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True, trust_remote_code=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=49)
    batch_size=12
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    latencies = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            start = time.time()
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).cuda()
            outputs = model(**inputs)
            end = time.time()
            latencies.append(end-start)

    
    if rank == 0:
        if not os.path.exists(DIR):
            os.makedirs(DIR, exist_ok=True)

        with open(f"{DIR}/e2e.csv", "w") as f:
            fieldnames = ["Iteration Number", "Latency (s)"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, latency in enumerate(latencies):
                writer.writerow({"Iteration Number": idx, "Latency (s)": latency})

    cleanup()

def __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python3 start.py num_gpus port_number")
        exit(1)

    world_size = int(sys.argv[1])
    port = sys.argv[2]
    mp.spawn(run_training, args=(world_size, port, scheduling_policy), nprocs=world_size, join=True)