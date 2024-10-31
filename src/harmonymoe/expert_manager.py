import torch
import torch.nn as nn
import torch.distributed as dist
import copy
import random

class ExpertManager():
    def __init__(self, experts: nn.ModuleList, cache_size: int, dynamic_components: list, fixed_cache=None, reference_cache=None, cache_policy="MTU"):
        self.cpu_experts = experts
        self.num_experts = len(experts)
        self.rank = dist.get_rank()
        self.num_gpus = dist.get_world_size()
        self.cache_size = cache_size
        self.dynamic_components = dynamic_components
        self.fixed_cache = fixed_cache
        # Cache can only hold up to cache_size
        self.reference_cache = list(map(lambda x: x[:cache_size], reference_cache)) if reference_cache else None 
        self.cache_policy = cache_policy

        self.validate_arguments()

        self.executer = self.execute_job if self.fixed_cache is None else self.direct_execute_job
        self.rng = random.Random(32)
        self.cache = [[None for _ in range(self.cache_size)] for _ in range(self.num_gpus)]

        self.stream = torch.cuda.Stream()

        # Duplicate an expert cache_size times onto GPU
        self.cached_experts = [copy.deepcopy(experts[0]).cuda() for _ in range(self.cache_size)]
        self.is_slot_loaded = [torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)]
    
    def validate_arguments(self):
        assert self.num_experts > 0, "There must be atleast 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."
        assert len(self.dynamic_components) > 0, "Expert has no dynamic components."
        if self.fixed_cache is not None:
            assert len(self.fixed_cache[self.rank]) <= self.cache_size, "When specifying fixed_cache need to ensure cache_size is big enough."

        match self.cache_policy:
            case "RAND": 
                self.gen_cache = self.gen_random_cache
            case "MRU":
                self.gen_cache = self.gen_most_recently_used_cache
            case "MTU":
                self.gen_cache = self.gen_most_tokens_used_cache
            case _:
                self.gen_cache = self.gen_same_cache

    def cuda(self, device=None):
        if self.fixed_cache:
            new_cache = self.fixed_cache[:]
        elif self.reference_cache:
            new_cache = self.reference_cache[:]
        else:
            new_cache = self.gen_random_cache(None) # Just gen randomly

        self.known_cache = new_cache
        self.load_cache(new_cache)

    def __call__(self, workload: list, schedule=None):
        return self.executer(workload, schedule=schedule)
        
    def load_cache(self, new_cache):
        new_rank_cache = new_cache[self.rank]
        new_rank_cache_aligned = [None for _ in range(self.cache_size)]

        loc = 0
        for expert in new_rank_cache:
            if expert in self.cache[self.rank]:
                # First need to check if there is something already there, if so need to move it
                idx = self.cache[self.rank].index(expert)
                if new_rank_cache_aligned[idx] != None:
                    orphan_expert = new_rank_cache_aligned[idx]
                    while new_rank_cache_aligned[loc] != None:
                        loc += 1
                    new_rank_cache_aligned[loc] = orphan_expert
                    loc += 1
                new_rank_cache_aligned[idx] = expert

                # new_rank_cache_aligned[self.cache[self.rank].index(expert)] = expert
            else:
                while new_rank_cache_aligned[loc] != None:
                    loc += 1
                new_rank_cache_aligned[loc] = expert
                loc += 1
        
        for slot_idx, expert_idx in enumerate(new_rank_cache_aligned):
            if expert_idx is not None:
                self.load_expert(expert_idx, slot_idx)

        for i in range(self.num_gpus):
            if i != self.rank:
                self.cache[i] = new_cache[i]        
    
    def get_cached(self):
        return copy.deepcopy(self.known_cache)
    
    def load_expert(self, expert_idx: int, load_idx: int): 
        if self.cache[self.rank][load_idx] == expert_idx:
            return
        self.cache[self.rank][load_idx] = expert_idx
        with torch.no_grad():
            with torch.cuda.stream(self.stream):
                for component in self.dynamic_components:
                    getattr(self.cached_experts[load_idx], component).weight.copy_(getattr(self.cpu_experts[expert_idx], component).weight)
                self.is_slot_loaded[load_idx].record()
    
    # Returns index of expert.
    def expert_loaded_location(self, expert_idx):
        idx = -1
        for i, v in enumerate(self.cache[self.rank]):
            if v == expert_idx:
                idx = i
                break
        if idx == -1:
            raise Exception(f"Expert {expert_idx} is not cached on rank {self.rank}.")
        return idx 
    
    def is_expert_loaded(self, expert_idx):
        return expert_idx in self.cache[self.rank]
    
    def direct_execute_job(self, workload: list, schedule=None):
        sizes = list(map(lambda x: x.size(dim=0), workload))
        experts_to_execute = list(filter(lambda e: sizes[e] != 0, range(self.num_experts)))
        for expert_idx in range(self.num_experts):
            if workload[expert_idx].size(dim=0) == 0:
                continue
            slot_idx = self.expert_loaded_location(expert_idx)
            workload[expert_idx] = self.cached_experts[slot_idx](workload[expert_idx])
        
        return workload
    
    def execute_job(self, workload: list, schedule = None):
        expert_order = []
        num_execs = 0

        # Setup
        for expert_idx in range(self.num_experts):
            if workload[expert_idx].size(dim=0) == 0:
                continue
            
            num_execs += 1
            if self.is_expert_loaded(expert_idx):
                expert_order.insert(0, expert_idx)
            else:
                expert_order.append(expert_idx)

        unused_expert_slots = []
        # Start loading as many experts as possible
        for slot_idx, expert_idx in enumerate(self.cache[self.rank]):
            if expert_idx is None:
                next_expert_idx = self.next_not_loaded_expert(expert_order)
                if next_expert_idx is None:
                    break
                self.load_expert(next_expert_idx, slot_idx)
            elif expert_idx not in expert_order:
                unused_expert_slots.append(slot_idx)
       
        # Fill every unused expert
        for slot_idx in unused_expert_slots:
            next_expert_idx = self.next_not_loaded_expert(expert_order)
            if next_expert_idx == None:
                break
            self.load_expert(next_expert_idx, slot_idx)
        
        # Begin execution
        for idx, expert_idx in enumerate(expert_order):
            slot_idx = self.expert_loaded_location(expert_idx)
            self.is_slot_loaded[slot_idx].record()
            workload[expert_idx] = self.cached_experts[slot_idx](workload[expert_idx])
            # Check if anything else needs loading
            next_expert_idx = self.next_not_loaded_expert(expert_order, idx)
            if next_expert_idx is not None:
                self.load_expert(next_expert_idx, slot_idx)
        
        self.update_cache(schedule)
        
        return workload

        
    def next_not_loaded_expert(self, experts, start=0):
        for idx in range(start,len(experts)):
            if experts[idx] not in self.cache[self.rank]:
                return experts[idx]
        return None

    def update_cache(self, schedule):
        if schedule == None:
            return # Do not update cache 
        
        if self.fixed_cache:
            return # Do not update a fixed cache
        
        new_cache = self.gen_cache(schedule)

        if self.reference_cache:
            temp = self.reference_cache[:]
            for expert_idx in new_cache[self.rank]:
                if len(temp[self.rank]) >= self.cache_size:
                    break
                if expert_idx not in temp[self.rank]:
                    temp[self.rank].append(expert_idx)
            new_cache = temp
        
        self.known_cache = new_cache
        self.load_cache(new_cache)
    
    def gen_same_cache(self, _):
        return self.cache
    
    def gen_random_cache(self, _):
        expert_idxs = list(range(self.num_experts))
        self.rng.shuffle(expert_idxs)
        new_cache = [expert_idxs[i:i+self.cache_size] for i in range(0, self.cache_size*self.num_gpus, self.cache_size)]
        return new_cache
    
    def gen_most_tokens_used_cache(self, schedule):
        new_cache = [[] for _ in range(self.num_gpus)]

        for i in range(self.num_gpus):
            expert_tokens = [0 for _ in range(self.num_experts)]
            for j in range(self.num_experts):
                for k in range(self.num_gpus):
                    expert_tokens[j] += schedule[k][j][i]

            experts_most_toks = map(lambda x: (x, expert_tokens[x]), range(self.num_experts))
            experts_most_toks = filter(lambda x: x[1] != 0, experts_most_toks)
            experts_most_toks = list(sorted(experts_most_toks, key=lambda x: x[1]))
            first_cache_size_experts_most_toks = list(map(lambda x: x[0], experts_most_toks[:self.cache_size]))
            
            new_cache[i] = first_cache_size_experts_most_toks
        
        return new_cache

    def gen_most_recently_used_cache(self, schedule):
        # For this need to generate the work order
        pass 