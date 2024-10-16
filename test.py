from scheduleatrix.utils import replace_moe_layer
from scheduleatrix.moe_layer import MoEConfig

from transformers import AutoModel
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch

def start(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2343"
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    dist.init_process_group("nccl", rank=rank, world_size=1)
    torch.cuda.set_device(rank)

    model = AutoModel.from_pretrained("google/switch-base-8", cache_dir="/cache")
    
    config = MoEConfig(
        expert_cache_size=8,
        dynamic_components=["wi", "wo"]
    )
    moe_layers = replace_moe_layer(
        model, 
        "SwitchTransformersSparseMLP", 
        "router", 
        "experts",
        "decoder",
        config,
    )

if __name__ == "__main__":
    mp.spawn(start, nprocs=1, join=True)