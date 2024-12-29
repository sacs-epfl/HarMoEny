import torch
import torch.nn as nn
import torch.distributed as dist
import copy
import random
import threading

from . import fetching_strategies
from .fetching_strategies import FetchingStrategyConfig


class ExpertManager:
    def __init__(
        self,
        experts: nn.ModuleList,
        cache_size: int,
        cache=None,
        fetching_strategy="async-cpu",
        layer_idx=-1,
    ):
        self.experts = experts
        self.num_experts = len(experts)
        self.cache = cache  # Get rid of cache and just use start and end, assume that experts are incrementing order
        self.cache_size = cache_size
        self.layer_idx = layer_idx
        self.rank = dist.get_rank()
        self.fetching_strategy = fetching_strategy

        # self.disable_async_fetch = disable_async_fetch
        # self.gpu_fetch = gpu_fetch

        self.validate_arguments()

        self.rng = random.Random(32)

        # self.num_swaps = 0
        # self.num_swaps_iters = []

        self.world_size = dist.get_world_size()
        self.num_experts_per_gpu = self.num_experts // self.world_size
        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

        # self.first_slot_expert_idx = self.cache[self.rank][0]
        # self.last_slot_expert_idx = self.cache[self.rank][-1]

        self.cached_experts = [
            copy.deepcopy(self.experts[0]).cuda() for _ in range(self.cache_size)
        ]
        self.buffer_expert = copy.deepcopy(self.experts[0]).cuda() if self.fetching_strategy == "gpu" else None
        # self.is_slot_loaded = [
        #     torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)
        # ]
        # self.slot_finished_executing_events = [
        #     torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)
        # ]

        self.expert_loaded_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)
        ]
        # self.expert_finished_executing_events = [
        #     torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)
        # ]

        for slot_idx, expert_idx in enumerate(self.cache[self.rank]):
            self.load_expert_into_slot(expert_idx, slot_idx)

        config = FetchingStrategyConfig(
            rank=self.rank,
            world_size=self.world_size,
            experts=self.experts,
            cache=self.cache,
            cached_experts=self.cached_experts,
            buffer_expert=self.buffer_expert,
            expert_loaded_events=self.expert_loaded_events,
            cache_size=self.cache_size,
            num_experts=self.num_experts,
            first_slot_expert_idx=self.cache[self.rank][0],
            last_slot_expert_idx=self.cache[self.rank][-1],
        )

        if fetching_strategy == "async-cpu":
            self.executor = fetching_strategies.AsynchronousCPU(config)
        elif fetching_strategy == "sync-cpu":
            self.executor = fetching_strategies.SynchronousCPU(config)
        elif fetching_strategy == "gpu":
            self.executor = fetching_strategies.GPU(config)
        else:
            raise NotImplementedError(f"Strategy {fetching_strategy} is not implemented")

        # self.executor = (
        #     self.execute_job
        #     if not self.disable_async_fetch
        #     else self.execute_job_no_async
        # )

    # def get_statistics(self):
    #     return {"number expert swaps": self.num_swaps_iters[:]}

    def load_expert_into_slot(self, expert_idx, slot_idx):
        with torch.no_grad():
            pinned_state_dict = self.experts[expert_idx].state_dict()
            cached_expert = self.cached_experts[slot_idx]
            for name, param in cached_expert.named_parameters():
                cpu_param = pinned_state_dict[name]
                param.copy_(cpu_param, non_blocking=False)

    def validate_arguments(self):
        assert self.num_experts > 0, "There must be atleast 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."

    def __call__(self, tokens, expert_mask, schedule=None):
        return self.executor.execute_job(tokens, expert_mask, schedule=schedule)
