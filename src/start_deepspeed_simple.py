# https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf
# https://github.com/microsoft/DeepSpeed

import torch
import torch.nn as nn
import sys
import os 
import time
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import argparse 
import deepspeed

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int) 
parser.add_argument("--world_size", default=8, type=int)
args = parser.parse_args()

def setup():
    os.environ["HF_HOME"] = "/cache"
    os.environ["HF_DATASETS_CACHE"] = "/cache"
    os.environ["TRITON_HOME"] = "/.triton"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)

def run_inference_workload():
    setup()

    model_name = f"google/switch-base-128"
    model = AutoModel.from_pretrained(model_name, cache_dir="/cache")

    class MLPWrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child
        
        def forward(self, x):
            x = self.child(x)
            return x[0], (x[1], x[2])

    # Update to add DeepspeedMoE to it
    def add_deepspeed_moe_model(module):
        if type(module).__name__ == "SwitchTransformersLayerFF":
            for child_name, child in module.named_children():
                if type(child).__name__ == "SwitchTransformersSparseMLP":
                    router = getattr(child, "router")
                    experts = getattr(child, "experts")
                    if type(experts) == nn.ModuleDict:
                        experts = list(experts.values())
                    
                    num_experts_per_gpu = 128 // args.world_size

                    experts = experts[args.local_rank*num_experts_per_gpu:(args.local_rank+1)*num_experts_per_gpu]

                    new = deepspeed.moe.layer.MoE(
                        hidden_size=768,
                        expert=experts[0],
                        num_experts=128,
                        ep_size=args.world_size,
                        k=1,
                        #eval_capacity_factor=54.65,
                        drop_tokens=False,
                        use_tutel=True,
                        top2_2nd_expert_sampling=False,
                        use_rts=False,
                    )

                    with torch.no_grad():
                        new.deepspeed_moe.gate.wg.weight.copy_(router.classifier.weight)
                        for i in range(len(experts)):
                            new.deepspeed_moe.experts.deepspeed_experts[i].wi.weight.copy_(experts[i].wi.weight)
                            new.deepspeed_moe.experts.deepspeed_experts[i].wo.weight.copy_(experts[i].wo.weight)

                    setattr(module, child_name, MLPWrapper(new))
        else:
            for child in module.children():
                add_deepspeed_moe_model(child)

    add_deepspeed_moe_model(model)

    model.eval()
    model.cuda()

    ds_engine = deepspeed.init_inference(
        model,
        dtype=torch.float,
        replace_with_kernel_inject=False,
        moe={
            "enabled": True,
            "ep_size": args.world_size, 
            "moe_experts": [128],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cache")

    dataset = load_dataset(
        "bookcorpus/bookcorpus", 
        split=f"train[:10400]", 
        streaming=False, 
        trust_remote_code=True, 
        cache_dir="/cache"
    )
    sampler = DistributedSampler(
        dataset, 
        num_replicas=args.world_size, 
        rank=args.local_rank, 
        shuffle=True, 
        seed=49
    )
    def collate_fn(batch):
        texts = ["summarize: " + item["text"] for item in batch]
        
        tokenized = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=120,
            return_tensors="pt"
        )
        return {
            **tokenized,
            "decoder_input_ids": torch.tensor([[tokenizer.pad_token_id]]*len(batch))
        }
    loader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=150,
        collate_fn=collate_fn,
    )

    run_standard_experiment(ds_engine, loader)


def run_standard_experiment(ds_engine, loader):    
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            ds_engine(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
            )

if __name__ == "__main__":
    run_inference_workload()

    print("All done :)")