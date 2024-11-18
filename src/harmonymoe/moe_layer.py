from .scheduler import Scheduler
from .expert_manager import ExpertManager

import torch.nn as nn
import torch
import torch.distributed as dist 

from dataclasses import dataclass
import csv

@dataclass
class MoEConfig:
    layer_idx: int = None
    is_decoder: bool = False
    scheduling_policy: str = "adnexus"
    cache_policy: str = "RAND"
    expert_cache_size: int = None
    dynamic_components: any = None
    eq_tokens: int = 150
    d_model: int = 768

class MoELayer(nn.Module):
    def __init__(self, router, experts, config=MoEConfig):

        super(MoELayer, self).__init__()

        self.router = router 

        self.layer_idx = config.layer_idx
        self.is_decoder = config.is_decoder

        self.num_experts = len(experts)
        self.num_gpus = dist.get_world_size()
        self.rank = dist.get_rank()

        # For statistics 
        self.tot_num_toks_send = []
        self.tot_num_toks_recv = []
        self.latencies = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.computation_latencies = []
        self.comp_start_event = torch.cuda.Event(enable_timing=True)
        self.comp_end_event = torch.cuda.Event(enable_timing=True)

        self.expert_freqs = []

        # Create our scheduler and exper manager
        self.scheduler = Scheduler(
            scheduling_policy=config.scheduling_policy if not self.is_decoder else "deepspeed", 
            num_experts=self.num_experts,
            eq_tokens=config.eq_tokens,
            d_model=config.d_model,
        )

        self.expert_manager = ExpertManager(
            experts, 
            config.expert_cache_size,
            config.dynamic_components,
            fixed_cache=self.scheduler.get_fixed_assign(),
            reference_cache=self.scheduler.get_reference_assign(),
            cache_policy=config.cache_policy,
        )
    
    # This is called on all modules to apply certain functions such as cuda
    def _apply(self, fn):
        fn(self.router)
        fn(self.expert_manager)
        return self

    def save_statistics(self, DIR=""):
        path = f"{DIR}/moe"
        if self.is_decoder:
            path += "_decoder"
        path += f"_layer-{self.layer_idx}"

        with open(f"{path}.csv", "w") as f:
            fieldnames = ["iteration", "total number of tokens sent", "total number of tokens recv", "latency (ms)", "comp latency (ms)", "expert distribution"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.tot_num_toks_send)):
                dic = {
                    "iteration": i,
                    "latency (ms)": self.latencies[i],
                    "comp latency (ms)": self.computation_latencies[i],
                    "total number of tokens sent": self.tot_num_toks_send[i],
                    "total number of tokens recv": self.tot_num_toks_recv[i],
                    "expert distribution": self.expert_freqs[i],
                }
             
                writer.writerow(dic)        

    @torch.no_grad()
    def forward(self, hidden_states):
        self.start_event.record()

        router_mask, router_probs, router_logits = self.router(hidden_states)
        # router_mask has dim (batch_size, seq_len, num_experts)
        # Entry will be a 1 on which expert to work on for the specific token
        # at specific sequence index on specific sample, rest will be 0

        self.expert_freqs.append(router_mask.sum(dim=(0,1)).tolist())
        
        expert_index = torch.argmax(router_mask, dim=-1)

        router_mask = router_mask.bool()

        num_toks_per_expert = []

        # Collect some stats
        tot = 0
        for j in range(self.num_experts):
            size = hidden_states[router_mask[:,:,j]].shape[0]
            num_toks_per_expert.append(size)
            tot += size
        self.tot_num_toks_send.append(tot)
        
        metadata_send = [torch.tensor(num_toks_per_expert, dtype=torch.int, device="cuda") for _ in range(self.num_gpus)]
        metadata_recv = [torch.zeros(self.num_experts, dtype=torch.int, device="cuda") for _ in range(self.num_gpus)]

        # Metadata all_to_all
        dist.all_to_all(metadata_recv, metadata_send)

        # Create global schedule
        schedule = self.scheduler(metadata_recv, self.expert_manager.get_cached())

        # Turn schedule and hidden_states into array of tensors
        # to distribute to each GPU
        tokens_send = self.scheduler.distribute_tokens(schedule, hidden_states, router_mask)
        tokens_recv = self.scheduler.allocate_recv_tensors(schedule)
        self.tot_num_toks_recv.append(sum(list(map(lambda x: x.shape[0], tokens_recv))))
        
        dist.all_to_all(tokens_recv, tokens_send)

        expert_tokens = self.scheduler.group_experts(schedule, tokens_recv)
        self.comp_start_event.record()
        expert_tokens = self.expert_manager(expert_tokens, schedule=schedule)
        self.comp_end_event.record()
        self.comp_end_event.synchronize()
        self.computation_latencies.append(self.comp_start_event.elapsed_time(self.comp_end_event))
        tokens_recv = self.scheduler.ungroup_experts(schedule, expert_tokens)

        dist.all_to_all(tokens_send, tokens_recv)

        hidden_states = self.scheduler.gather_tokens(schedule, tokens_send, hidden_states, router_mask)

        self.end_event.record()
        self.end_event.synchronize()
        self.latencies.append(self.start_event.elapsed_time(self.end_event))

        hidden_states = router_probs * hidden_states
        return hidden_states, (router_logits, expert_index)