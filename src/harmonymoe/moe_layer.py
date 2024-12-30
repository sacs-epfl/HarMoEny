from .scheduler import Scheduler
from .expert_manager import ExpertManager
from .router import RouterConfig, Router

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist

from dataclasses import dataclass
import csv


@dataclass
class MoEConfig:
    experts: list[any] = None
    router_weights: any = None
    layer_idx: int = None
    scheduling_policy: str = "adnexus"
    cache_policy: str = "RAND"
    expert_cache_size: int = None
    eq_tokens: int = 150
    d_model: int = 768
    world_size: int = 1
    expert_placement: list = None
    fetching_strategy: str = "async-cpu"
    model_name: str = None
    num_experts: int = 8
    enable_skew: bool = False
    enable_random: bool = False
    enable_uniform: bool = False
    skew: int = 0.05
    num_experts_skewed: int = 1


class MoELayer(nn.Module):
    def __init__(self, config=MoEConfig):
        super(MoELayer, self).__init__()

        self.router = Router(
            RouterConfig(
                d_model=config.d_model,
                num_experts=config.num_experts,
                weights=config.router_weights,
                enable_skew=config.enable_skew,
                enable_random=config.enable_random,
                enable_uniform=config.enable_uniform,
                skew=config.skew,
                num_experts_skewed=config.num_experts_skewed,
            )
        )

        self.config = config
        self.layer_idx = config.layer_idx
        self.num_experts = config.num_experts
        self.num_gpus = config.world_size
        self.d_model = config.d_model

        # For statistics
        self.tot_num_toks_send = []
        self.tot_num_toks_recv = []

        self.latencies = []
        self.metadata_latencies = []
        self.schedule_latencies = []
        self.first_transfer_latencies = []
        self.second_transfer_latencies = []
        self.computation_latencies = []

        self.expert_freqs = []

    def prepare(self):
        self.rank = dist.get_rank()

        self.start_pass = torch.cuda.Event(enable_timing=True)
        self.end_pass = torch.cuda.Event(enable_timing=True)
        self.start_metadata = torch.cuda.Event(enable_timing=True)
        self.end_metadata = torch.cuda.Event(enable_timing=True)
        self.start_schedule = torch.cuda.Event(enable_timing=True)
        self.end_schedule = torch.cuda.Event(enable_timing=True)
        self.start_first_transfer = torch.cuda.Event(enable_timing=True)
        self.end_first_transfer = torch.cuda.Event(enable_timing=True)
        self.start_computation = torch.cuda.Event(enable_timing=True)
        self.end_computation = torch.cuda.Event(enable_timing=True)
        self.start_second_transfer = torch.cuda.Event(enable_timing=True)
        self.end_second_transfer = torch.cuda.Event(enable_timing=True)

        self.scheduler = Scheduler(
            scheduling_policy=self.config.scheduling_policy,
            num_experts=self.num_experts,
            eq_tokens=self.config.eq_tokens,
            d_model=self.d_model,
            num_gpus=self.num_gpus,
            expert_placement=self.config.expert_placement,
            layer_idx=self.layer_idx,
        )

        self.expert_manager = ExpertManager(
            self.config.experts,
            self.config.expert_cache_size,
            cache=self.scheduler.get_cache(),
            fetching_strategy=self.config.fetching_strategy,
            layer_idx=self.layer_idx,
        )

    def get_statistics(self):
        stats = []
        for i in range(len(self.tot_num_toks_send)):
            dic = {
                "latency (ms)": self.latencies[i],
                "metadata latency (ms)": (
                    self.metadata_latencies[i]
                    if i < len(self.metadata_latencies)
                    else -1
                ),
                "schedule latency (ms)": (
                    self.schedule_latencies[i]
                    if i < len(self.schedule_latencies)
                    else -1
                ),
                "first transfer latency (ms)": (
                    self.first_transfer_latencies[i]
                    if i < len(self.first_transfer_latencies)
                    else -1
                ),
                "second transfer latency (ms)": (
                    self.second_transfer_latencies[i]
                    if i < len(self.second_transfer_latencies)
                    else -1
                ),
                "comp latency (ms)": (
                    self.computation_latencies[i]
                    if i < len(self.computation_latencies)
                    else -1
                ),
                "total number of tokens sent": self.tot_num_toks_send[i],
                "total number of tokens recv": self.tot_num_toks_recv[i],
                "expert distribution": self.expert_freqs[i],
            }

            stats.append(dic)
        return stats

    @torch.no_grad()
    def forward(self, hidden_states):
        self.start_pass.record()

        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.config.d_model)

        router_mask, router_probs, router_logits = self.router(hidden_states)
        # router_mask has dim (batch_size, seq_len, num_experts)
        # Entry will be a 1 on which expert to work on for the specific token
        # at specific sequence index on specific sample, rest will be 0

        expert_index = torch.argmax(router_mask, dim=-1)
        router_mask = router_mask.bool()

        expert_indices = [
            router_mask[:, j].nonzero(as_tuple=True) for j in range(self.num_experts)
        ]

        num_toks_per_expert = [
            expert_indices[j][0].shape[0] for j in range(self.num_experts)
        ]
        self.expert_freqs.append(num_toks_per_expert)

        metadata_send = torch.tensor(
            [num_toks_per_expert] * self.num_gpus, dtype=torch.int, device="cuda"
        )
        metadata_recv = torch.zeros(
            (self.num_gpus, self.num_experts), dtype=torch.int, device="cuda"
        )

        # Metadata all_to_all_single
        self.start_metadata.record()
        dist.all_to_all_single(metadata_recv, metadata_send)
        self.end_metadata.record()

        # Create global schedule
        self.start_schedule.record()
        schedule = self.scheduler(metadata_recv)
        schedule_list = schedule.tolist()
        self.end_schedule.record()

        num_toks_send = schedule[self.rank].sum()
        self.tot_num_toks_send.append(num_toks_send.item())

        # If dropping need a way to make update hidden_states passed to distrbute_tokens
        tokens_send, send_splits = self.scheduler.distribute_tokens(
            schedule_list, hidden_states, expert_indices, num_toks_send
        )
        tokens_recv, recv_splits = self.scheduler.allocate_recv_tensors(schedule)
        total_tokens = tokens_recv.shape[0]
        self.tot_num_toks_recv.append(total_tokens)

        self.start_first_transfer.record()
        dist.all_to_all_single(
            tokens_recv,
            tokens_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        self.end_first_transfer.record()

        expert_mask = self.scheduler.generate_expert_mask(schedule, total_tokens)
        self.start_computation.record()
        self.expert_manager(tokens_recv, expert_mask, schedule=schedule)
        self.end_computation.record()

        self.start_second_transfer.record()
        dist.all_to_all_single(
            tokens_send,
            tokens_recv,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )
        self.end_second_transfer.record()

        hidden_states = self.scheduler.gather_tokens(
            schedule_list, tokens_send, hidden_states, expert_indices
        )

        self.end_pass.record()
        self.end_pass.synchronize()  # Will only have a single synchronize on the final event
        # Collect data
        self.latencies.append(self.start_pass.elapsed_time(self.end_pass))
        self.metadata_latencies.append(
            self.start_metadata.elapsed_time(self.end_metadata)
        )
        self.schedule_latencies.append(
            self.start_schedule.elapsed_time(self.end_schedule)
        )
        self.first_transfer_latencies.append(
            self.start_first_transfer.elapsed_time(self.end_first_transfer)
        )
        self.computation_latencies.append(
            self.start_computation.elapsed_time(self.end_computation)
        )
        self.second_transfer_latencies.append(
            self.start_second_transfer.elapsed_time(self.end_second_transfer)
        )

        hidden_states = router_probs * hidden_states
        hidden_states = hidden_states.view(original_shape)

        return hidden_states, (router_logits, expert_index)
