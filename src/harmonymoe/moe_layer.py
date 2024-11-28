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
    scheduling_policy: str = "adnexus"
    cache_policy: str = "RAND"
    expert_cache_size: int = None
    dynamic_components: any = None
    eq_tokens: int = 150
    d_model: int = 768
    world_size: int = 1
    expert_placement: list = None

class MoELayer(nn.Module):
    def __init__(self, router, experts, config=MoEConfig):

        super(MoELayer, self).__init__()

        self.router = router 

        self.layer_idx = config.layer_idx

        self.num_experts = len(experts)
        self.num_gpus = config.world_size

        # For statistics 
        self.tot_num_toks_send = []
        self.tot_num_toks_recv = []
        self.latencies = []
        self.metadata_comm_latencies = []
        self.computation_latencies = []

        self.expert_freqs = []

        # Create our scheduler and exper manager
        self.scheduler = Scheduler(
            scheduling_policy=config.scheduling_policy, 
            num_experts=self.num_experts,
            eq_tokens=config.eq_tokens,
            d_model=config.d_model,
            num_gpus=self.num_gpus,
            expert_placement=config.expert_placement,
            layer_idx=self.layer_idx,
        )

        self.expert_manager = ExpertManager(
            experts, 
            config.expert_cache_size,
            config.dynamic_components,
            fixed_cache=self.scheduler.get_fixed_assign(),
            reference_cache=self.scheduler.get_reference_assign(),
            cache_policy=config.cache_policy,
            num_gpus=self.num_gpus,
        )
    
    # This is called on all modules to apply certain functions such as cuda
    def _apply(self, fn):
        try:
            fn(self.router)
        except:
            pass
        try:
            fn(self.expert_manager)
        except Exception as e:
            print(f"Error applying function '{repr(fn)}' to 'expert_manager': {e}")

        return self

    def get_statistics(self, DIR=""):
        stats = []
        for i in range(len(self.tot_num_toks_send)):
            stats.append({
                "latency (ms)": self.latencies[i],
                "metadata latency (ms)": self.metadata_comm_latencies[i] if i < len(self.metadata_comm_latencies) else -1,
                "comp latency (ms)": self.computation_latencies[i] if i < len(self.computation_latencies) else -1,
                "total number of tokens sent": self.tot_num_toks_send[i],
                "total number of tokens recv": self.tot_num_toks_recv[i],
                "expert distribution": self.expert_freqs[i],
            })
        return stats

             

    @torch.no_grad()
    def forward(self, hidden_states):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        router_mask, router_probs, router_logits = self.router(hidden_states)
        # router_mask has dim (batch_size, seq_len, num_experts)
        # Entry will be a 1 on which expert to work on for the specific token
        # at specific sequence index on specific sample, rest will be 0

        #self.expert_freqs.append(router_mask.sum(dim=(0,1)).tolist())
        
        expert_index = torch.argmax(router_mask, dim=-1)
        router_mask = router_mask.bool()

        num_toks_per_expert = []
        num_toks_send = 0
        for j in range(self.num_experts):
            size = hidden_states[router_mask[:,:,j]].shape[0]
            num_toks_per_expert.append(size)
            num_toks_send += size
        self.tot_num_toks_send.append(num_toks_send)
        self.expert_freqs.append(num_toks_per_expert)
        
        metadata_send = torch.tensor([num_toks_per_expert] * self.num_gpus, dtype=torch.int, device="cuda")
        metadata_recv = torch.zeros((self.num_gpus, self.num_experts), dtype=torch.int, device="cuda")

        # Metadata all_to_all_single
        dist.all_to_all_single(metadata_recv, metadata_send)
        
        # Create global schedule
        schedule = self.scheduler(metadata_recv, self.expert_manager.get_cached())

        # Turn schedule and hidden_states into array of tensors
        # to distribute to each GPU
        tokens_send, send_splits = self.scheduler.distribute_tokens(schedule, hidden_states, router_mask, num_toks_send)
        tokens_recv, recv_splits = self.scheduler.allocate_recv_tensors(schedule)
        self.tot_num_toks_recv.append(tokens_recv.shape[0])

        # tokens_send = [t.contiguous() for t in tokens_send]
        # tokens_send_tensor = torch.cat(tokens_send, dim=0)
        #print(f"[Rank:{dist.get_rank()}] {tokens_send_tensor.shape}")
        #send_splits = [t.shape[0] for t in tokens_send]

        #recv_splits = [t.shape[0] for t in tokens_recv]
        #tokens_recv_tensor = torch.empty((sum(recv_splits), tokens_send.shape[1]), device="cuda")
        #print(f"[Rank:{dist.get_rank()}] {tokens_recv_tensor.shape}")
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        #dist.all_to_all(tokens_recv, tokens_send)
        dist.all_to_all_single(
            tokens_recv,
            tokens_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        # end.record()
        # end.synchronize()
        # self.metadata_comm_latencies.append(start.elapsed_time(end))

        #tokens_recv = torch.split(tokens_recv_tensor, recv_splits)

        expert_tokens = self.scheduler.group_experts(schedule, tokens_recv)
        # comp_start_event = torch.cuda.Event(enable_timing=True)
        # comp_end_event = torch.cuda.Event(enable_timing=True)
        # comp_start_event.record()
        expert_tokens = self.expert_manager(expert_tokens, schedule=schedule)
        # comp_end_event.record()
        # comp_end_event.synchronize()
        # self.computation_latencies.append(comp_start_event.elapsed_time(comp_end_event))
        tokens_recv = self.scheduler.ungroup_experts(schedule, expert_tokens, tokens_recv.shape[0])

        dist.all_to_all_single(
            tokens_send, 
            tokens_recv, 
            output_split_sizes=send_splits, 
            input_split_sizes=recv_splits
        )

        hidden_states = self.scheduler.gather_tokens(schedule, tokens_send, hidden_states, router_mask)

        end_event.record()
        end_event.synchronize()
        self.latencies.append(start_event.elapsed_time(end_event))

        hidden_states = router_probs * hidden_states
        return hidden_states, (router_logits, expert_index)