import torch
import torch.nn as nn
import torch.distributed as dist
import copy
import random
import weakref
import nvtx
import threading


class ExpertManager:
    def __init__(
        self,
        experts: nn.ModuleList,
        cache_size: int,
        cache=None,
        disable_async_fetch=False,
        gpu_fetch=False,
        layer_idx=-1,
    ):
        self.experts = experts
        self.total_param_size = sum(p.numel() for p in self.experts[0].parameters())
        self.num_experts = len(experts)
        self.cache = cache  # Get rid of cache and just use start and end, assume that experts are incrementing order
        self.cache_size = cache_size
        self.layer_idx = layer_idx

        self.disable_async_fetch = disable_async_fetch
        self.gpu_fetch = gpu_fetch

        self.validate_arguments()

        self.executer = (
            self.execute_job
            if not self.disable_async_fetch
            else self.execute_job_no_async
        )
        self.rng = random.Random(32)

        self.num_swaps = 0
        self.num_swaps_iters = []

    def get_statistics(self):
        return {"number expert swaps": self.num_swaps_iters[:]}

    def validate_arguments(self):
        assert self.num_experts > 0, "There must be atleast 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."
        assert (
            self.disable_async_fetch and not self.gpu_fetch
        ), "You can only disable async for CPU offloading"

    def cuda(self, device=None):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_experts_per_gpu = self.num_experts // self.world_size
        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

        self.first_slot_expert_idx = self.cache[self.rank][0]
        self.last_slot_expert_idx = self.cache[self.rank][-1]
        self.work_order = self.generate_work_order()

        self.cached_experts = [
            copy.deepcopy(self.experts[0]).cuda() for _ in range(self.cache_size)
        ]
        if self.gpu_fetch:
            self.buffer_expert = copy.deepcopy(self.experts[0]).cuda()
        self.is_slot_loaded = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)
        ]
        self.slot_finished_executing_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)
        ]

        self.expert_loaded_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)
        ]
        self.expert_finished_executing_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)
        ]

        for slot_idx, expert_idx in enumerate(self.cache[self.rank]):
            self.load_expert_into_slot(expert_idx, slot_idx)

        return self

    def cpu(self):
        for cached_expert in self.cached_experts:
            cached_expert.cpu()

        self.load_stream = None
        self.comp_stream = None
        self.is_slot_loaded = None

        return self

    def __call__(self, tokens, expert_mask, schedule=None):
        return self.executer(tokens, expert_mask, schedule=schedule)

    ############# ASYNCHRONOUS CPU OFFLOAD ############
    def load_expert_into_slot(self, expert_idx, slot_idx):
        # with nvtx.annotate(f"Loading expert {expert_idx} into slot {slot_idx}", color="green"):
        with torch.no_grad():
            with torch.cuda.stream(self.load_stream):
                pinned_state_dict = self.experts[expert_idx].state_dict()
                cached_expert = self.cached_experts[slot_idx]
                for name, param in cached_expert.named_parameters():
                    cpu_param = pinned_state_dict[name]
                    param.copy_(cpu_param, non_blocking=True)
                self.expert_loaded_events[expert_idx].record(stream=self.load_stream)

    def execute_job(self, tokens, expert_mask, schedule=None):
        expert_order = self.cache[self.rank][:]

        # Setup
        for expert_idx in range(self.num_experts):
            if expert_mask[expert_idx].shape[0] != 0 and expert_idx not in expert_order:
                expert_order.append(expert_idx)

        # Rare instance will the cached expert have no tokens to process which will be just a no-op

        # Begin execution
        stale_slots = []
        for idx, expert_idx in enumerate(expert_order):
            slot_idx = idx % self.cache_size

            if expert_idx != self.cache[self.rank][slot_idx]:
                self.comp_stream.wait_event(self.expert_loaded_events[expert_idx])

            # with nvtx.annotate(f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"):
            with torch.cuda.stream(self.comp_stream):
                # Execute the expert on the tokens
                tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](
                    tokens[expert_mask[expert_idx]]
                )
                # Record that the expert has finished executing
            self.expert_finished_executing_events[expert_idx].record(
                stream=self.comp_stream
            )

            # Check if anything else needs loading
            if idx + self.cache_size < len(expert_order):
                stale_slots.append(slot_idx)
                self.load_stream.wait_event(
                    self.expert_finished_executing_events[expert_idx]
                )
                self.load_expert_into_slot(
                    expert_order[idx + self.cache_size], slot_idx
                )

        for stale_slot in stale_slots:
            self.load_expert_into_slot(self.cache[self.rank][stale_slot], stale_slot)

        return tokens

    ######## NO ASYNC ♯###############
    def load_expert_into_slot_no_async(self, expert_idx, slot_idx):
        # with nvtx.annotate(f"Loading expert {expert_idx} into slot {slot_idx}", color="green"):
        with torch.no_grad():
            pinned_state_dict = self.experts[expert_idx].state_dict()
            cached_expert = self.cached_experts[slot_idx]
            for name, param in cached_expert.named_parameters():
                cpu_param = pinned_state_dict[name]
                param.copy_(cpu_param, non_blocking=False)

    def execute_job_no_async(self, tokens, expert_mask, schedule=None):
        expert_order = self.cache[self.rank][:]
        cached_experts_with_no_tokens = []
        for expert_idx in range(self.num_experts):
            if expert_mask[expert_idx].shape[0] == 0:
                if expert_idx in self.cache[self.rank]:
                    cached_experts_with_no_tokens.append(expert_idx)
            else:
                if expert_idx not in self.cache[self.rank]:
                    expert_order.append(expert_idx)

        for expert_idx in cached_experts_with_no_tokens:
            expert_order.remove(expert_idx)

        loaded = False

        for idx, expert_idx in enumerate(expert_order):
            slot_idx = idx % self.cache_size

            if self.cache[self.rank][slot_idx] != expert_idx:
                # print(f"(rank:{self.rank}) loading expert: {expert_idx}")
                loaded = True
                slot_idx = 0
                event = torch.cuda.Event(enable_timing=False)
                self.load_expert_into_slot_no_async(expert_idx, slot_idx)
                event.record()
                event.synchronize()

            tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](
                tokens[expert_mask[expert_idx]]
            )

        if loaded:
            self.load_expert_into_slot_no_async(self.cache[self.rank][0], 0)

        return tokens

    ########### GPU FETCHING ##################
    def generate_work_order(self):
        experts = [i for i in range(self.num_experts)]
        start = self.first_slot_expert_idx
        end = self.last_slot_expert_idx
        work_order = experts[start : end + 1] + experts[:start] + experts[end + 1 :]
        return work_order

    def get_expert_processing_ranks(self, schedule):
        return schedule[
            :, self.first_slot_expert_idx : self.last_slot_expert_idx + 1, :
        ].sum(dim=0)

    def flatten_params(self, params):
        return torch.cat([p.view(-1) for p in params], dim=0)

    def unflatten_params(self, flattened, params):
        offset = 0
        for p in params:
            size = p.numel()
            p.data.copy_(flattened[offset : offset + size].view_as(p))
            offset += size

    def send_overloaded_weights(self, schedule):
        expert_processing_ranks = self.get_expert_processing_ranks(schedule)

        with torch.cuda.stream(self.load_stream):
            send_requests = []
            for j in range(self.cache_size):
                flattened = None
                for k in range(self.world_size):
                    if k == self.rank or expert_processing_ranks[j][k] == 0:
                        continue
                    if flattened is None:
                        flattened = self.flatten_params(
                            self.cached_experts[j].parameters()
                        )
                    req = dist.isend(flattened, dst=k)
                    send_requests.append(req)

            for req in send_requests:
                req.wait()  # Synchronize sends

    def request_expert_weights(self, expert_idx):
        with torch.no_grad():
            rank = expert_idx // self.num_experts_per_gpu

            flattened = torch.zeros(self.total_param_size, device="cuda")
            req = dist.irecv(flattened, src=rank)
            req.wait()  # Synchronize receive

            self.unflatten_params(flattened, self.buffer_expert.parameters())

    def execute_job_gpu_fetching(self, tokens, expert_mask, schedule=None):
        # Issue to comm stream the expert sends
        self.send_overloaded_weights(schedule)

        # Perform work
        for idx, expert_idx in enumerate(self.work_order):
            # Do we have work to do?
            if expert_mask[expert_idx].shape[0] == 0:
                continue

            # Is it a cached expert?
            if idx < self.cache_size:
                tokens[expert_mask[expert_idx]] = self.cached_experts[idx](
                    tokens[expert_mask[expert_idx]]
                )
            else:
                self.request_expert_weights(expert_idx)
                tokens[expert_mask[expert_idx]] = self.buffer_expert(
                    tokens[expert_mask[expert_idx]]
                )
