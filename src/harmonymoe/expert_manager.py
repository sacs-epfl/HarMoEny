import torch
import torch.nn as nn
import torch.distributed as dist
import copy
import random
import weakref

class ExpertManager():
    def __init__(self, experts: nn.ModuleList, cache_size: int, dynamic_components: list, fixed_cache=None, reference_cache=None, cache_policy="MTU", num_gpus=1, disable_async_fetch=False):
        self.experts = experts
        self.num_experts = len(experts)
        self.num_gpus = num_gpus
        self.cache_size = cache_size
        self.dynamic_components = dynamic_components
        self.fixed_cache = fixed_cache
        self.reference_cache = list(map(lambda x: x[:cache_size], reference_cache)) if reference_cache else None 
        self.cache_policy = cache_policy
        self.disable_async_fetch = disable_async_fetch

        self.validate_arguments()

        self.executer = self.execute_job if self.fixed_cache is None else self.direct_execute_job
        self.rng = random.Random(32)

        self.num_swaps = 0
        self.num_swaps_iters = []
    
    def get_statistics(self):
        return {
            "number expert swaps": self.num_swaps_iters[:]
        }
    
    def validate_arguments(self):
        assert self.num_experts > 0, "There must be atleast 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."
        assert len(self.dynamic_components) > 0, "Expert has no dynamic components."
        if self.fixed_cache is not None:
            assert len(self.fixed_cache[0]) <= self.cache_size, "When specifying fixed_cache need to ensure cache_size is big enough."

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
        self.stream = torch.cuda.Stream()
        self.cached_experts = [copy.deepcopy(self.experts[0]).cuda() for _ in range(self.cache_size)]
        self.is_slot_loaded = [torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)]

        self.cache = [[None for _ in range(self.cache_size)] for _ in range(self.num_gpus)]
        if self.fixed_cache:
            new_cache = self.fixed_cache[:]
        elif self.reference_cache:
            new_cache = self.reference_cache[:]
        else:
            new_cache = self.gen_random_cache(None) # Just gen randomly

        self.known_cache = new_cache
        self.load_cache(new_cache)

        return self

    def cpu(self):
        for cached_expert in self.cached_experts:
            cached_expert.cpu()
        
        self.stream = None
        self.is_slot_loaded = None 

        return self

    def __call__(self, tokens, expert_mask, schedule=None):
        return self.executer(tokens, expert_mask, schedule=schedule)
        
    def load_cache(self, new_cache):
        new_rank_cache = new_cache[dist.get_rank()]
        new_rank_cache_aligned = [None for _ in range(self.cache_size)]

        loc = 0
        for expert in new_rank_cache:
            if expert in self.cache[dist.get_rank()]:
                # First need to check if there is something already there, if so need to move it
                idx = self.cache[dist.get_rank()].index(expert)
                if new_rank_cache_aligned[idx] != None:
                    orphan_expert = new_rank_cache_aligned[idx]
                    while new_rank_cache_aligned[loc] != None:
                        loc += 1
                    new_rank_cache_aligned[loc] = orphan_expert
                    loc += 1
                new_rank_cache_aligned[idx] = expert
            else:
                while new_rank_cache_aligned[loc] != None:
                    loc += 1
                new_rank_cache_aligned[loc] = expert
                loc += 1
        
        for slot_idx, expert_idx in enumerate(new_rank_cache_aligned):
            if expert_idx is not None:
                self.load_expert(expert_idx, slot_idx)

        for i in range(self.num_gpus):
            if i != dist.get_rank():
                self.cache[i] = new_cache[i]        
    
    def get_cached(self):
        return copy.deepcopy(self.known_cache)
    
    def load_expert(self, expert_idx: int, load_idx: int): 
        if self.cache[dist.get_rank()][load_idx] == expert_idx:
            return
        self.cache[dist.get_rank()][load_idx] = expert_idx
        with torch.no_grad():
            if self.disable_async_fetch:
                for component in self.dynamic_components:
                    getattr(self.cached_experts[load_idx], component).weight.copy_(getattr(self.experts[expert_idx], component).weight)
                self.is_slot_loaded[load_idx].record()
            else:
                with torch.cuda.stream(self.stream):
                    for component in self.dynamic_components:
                        getattr(self.cached_experts[load_idx], component).weight.copy_(getattr(self.experts[expert_idx], component).weight)
                    self.is_slot_loaded[load_idx].record()
    
    # Returns index of expert.
    def expert_loaded_location(self, expert_idx):
        # Returns -1 if not found
        idx = -1
        for i, v in enumerate(self.cache[dist.get_rank()]):
            if v == expert_idx:
                idx = i
                break
        # if idx == -1:
        #     raise Exception(f"Expert {expert_idx} is not cached on rank {dist.get_rank()}.")
        return idx 
    
    def is_expert_loaded(self, expert_idx):
        return expert_idx in self.cache[dist.get_rank()]
    
    def direct_execute_job(self, tokens, expert_mask, schedule=None):
        for expert_idx in range(self.num_experts):
            if expert_mask[expert_idx].shape[0] == 0:
                continue
            slot_idx = self.expert_loaded_location(expert_idx)
            tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])
        
        return tokens
    
    def execute_job(self, tokens, expert_mask, schedule = None):
        expert_order = []
        num_execs = 0
        self.num_swaps = 0

        # Setup
        for expert_idx in range(self.num_experts):
            if expert_mask[expert_idx].shape[0] == 0:
                continue

            num_execs += 1
            if self.is_expert_loaded(expert_idx):
                expert_order.insert(0, expert_idx)
            else:
                expert_order.append(expert_idx)

        unused_expert_slots = []
        # Start loading as many experts as possible
        for slot_idx, expert_idx in enumerate(self.cache[dist.get_rank()]):
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
            # Check if not assigned to a cache which is possible if we have disabled async fetch
            if slot_idx == -1:
                # print("Need to fetch expert later")
                slot_idx = 0 # Just default to loading to the first slot
                self.num_swaps += 1
                self.load_expert(expert_idx, slot_idx) 

            self.is_slot_loaded[slot_idx].synchronize()  # Wait until the expert is loaded
            tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])

            # Check if anything else needs loading
            if not self.disable_async_fetch:
                next_expert_idx = self.next_not_loaded_expert(expert_order, idx)
                if next_expert_idx is not None:
                    self.num_swaps += 1
                    self.load_expert(next_expert_idx, slot_idx)
        
        self.update_cache(schedule)

        self.num_swaps_iters.append(self.num_swaps)
        
        return tokens

        
    def next_not_loaded_expert(self, experts, start=0):
        for idx in range(start,len(experts)):
            if experts[idx] not in self.cache[dist.get_rank()]:
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
            for expert_idx in new_cache[dist.get_rank()]:
                if len(temp[dist.get_rank()]) >= self.cache_size:
                    break
                if expert_idx not in temp[dist.get_rank()]:
                    temp[dist.get_rank()].append(expert_idx)
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