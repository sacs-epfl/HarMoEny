import torch
import torch.nn as nn
import torch.distributed as dist
import copy
import random
import weakref
import nvtx
import threading

class ExpertManager():
    def __init__(self, experts: nn.ModuleList, cache_size: int, dynamic_components: list, cache=None, disable_async_fetch=False, layer_idx=-1):
        self.experts = experts
        self.total_param_size = sum(p.numel() for p in self.experts[0].parameters())
        self.num_experts = len(experts)
        self.cache = cache # Get rid of cache and just use start and end, assume that experts are incrementing order
        self.cache_size = cache_size
        self.layer_idx = layer_idx
 
        self.dynamic_components = dynamic_components
        self.disable_async_fetch = disable_async_fetch

        self.validate_arguments()

        self.executer = self.execute_job_gpu_fetching if not self.disable_async_fetch else self.execute_job_no_async
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

    def cuda(self, device=None):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_experts_per_gpu = self.num_experts // self.world_size
        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()


        self.first_slot_expert_idx = self.cache[self.rank][0]
        self.last_slot_expert_idx = self.cache[self.rank][-1]
        self.work_order = self.generate_work_order()

       # self.slot_streams = [torch.cuda.Stream() for _ in range(self.cache_size)]

        self.cached_experts = [copy.deepcopy(self.experts[0]).cuda() for _ in range(self.cache_size)]
        self.buffer_expert = copy.deepcopy(self.experts[0]).cuda()
        self.is_slot_loaded = [torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)]
        self.slot_finished_executing_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.cache_size)]

        self.expert_loaded_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)]
        self.expert_finished_executing_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)]
               
        for slot_idx, expert_idx in enumerate(self.cache[self.rank]):
            self.load_expert_into_slot(expert_idx, slot_idx)

        #self.start_responder_thread()

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

            #with nvtx.annotate(f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"):
            with torch.cuda.stream(self.comp_stream):
            # Execute the expert on the tokens
                tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])
                # Record that the expert has finished executing
            self.expert_finished_executing_events[expert_idx].record(stream=self.comp_stream)

            # Check if anything else needs loading
            if idx + self.cache_size < len(expert_order):
                stale_slots.append(slot_idx)
                self.load_stream.wait_event(self.expert_finished_executing_events[expert_idx])
                self.load_expert_into_slot(expert_order[idx + self.cache_size], slot_idx)
  
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
                print(f"(rank:{self.rank}) loading expert: {expert_idx}")
                loaded = True
                slot_idx = 0
                event = torch.cuda.Event(enable_timing=False)
                self.load_expert_into_slot_no_async(expert_idx, slot_idx)
                event.record()
                event.synchronize()
            
            tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])

        if loaded:
            self.load_expert_into_slot_no_async(self.cache[self.rank][0], 0)
        
        return tokens


    ##### GPU FETCHING ############
    # def start_responder_thread(self):
    #     def responder_loop():
    #         request_tensor = torch.empty(1, dtype=torch.uint8, device="cuda")
    #         while True:
    #             for src_rank in range(self.world_size):
    #                 if src_rank == self.rank:
    #                     continue
                    
    #                 try:
    #                     dist.recv(request_tensor, src=src_rank, tag=self.layer_idx, timeout=0.001)
    #                 except RuntimeError:
    #                     continue # Go to the next

    #                 expert_idx = request_tensor.item()

    #                 with torch.no_grad():
    #                     for name, param in self.cached_experts[expert_idx-self.first_slot_expert_idx].named_parameters():
    #                         dist.send(param.data, dst=src_rank, tag=self.layer_idx)
            
    #     responder_thread = threading.Thread(target=responder_loop, daemon=True)
    #     responder_thread.start()
    #     return responder_thread
    
    def generate_work_order(self):
        experts = [i for i in range(self.num_experts)]
        start = self.first_slot_expert_idx
        end = self.last_slot_expert_idx
        work_order = experts[start:end+1] + experts[:start] + experts[end+1:]
        return work_order

    def get_expert_processing_ranks(self, schedule):
        return schedule[:, self.first_slot_expert_idx:self.last_slot_expert_idx+1, :].sum(dim=0)
    
    def flatten_params(self, params):
        return torch.cat([p.view(-1) for p in params], dim=0)

    def unflatten_params(self, flattened, params):
        offset = 0
        for p in params:
            size = p.numel()
            p.data.copy_(flattened[offset:offset + size].view_as(p))
            offset += size

    # def send_overloaded_weights(self, schedule):
    #     expert_processing_ranks = self.get_expert_processing_ranks(schedule)

    #     with torch.cuda.stream(self.load_stream):
    #         for j in range(self.cache_size):
    #             for k in range(self.world_size):
    #                 # Skip over ourselves-we are already loaded
    #                 if k == self.rank:
    #                     continue
    #                 # If rank not working on that expert, no need to send 
    #                 if expert_processing_ranks[j][k] == 0:
    #                     continue

    #                 for name, param in self.cached_experts[j].named_parameters():
    #                     dist.send(param.data, dst=k)

    # def request_expert_weights(self, expert_idx):
    #     with torch.no_grad():
    #         rank = expert_idx // self.num_experts_per_gpu

    #         # Receive the weights
    #         for name, param in self.buffer_expert.named_parameters():
    #             dist.recv(param.data, src=rank)
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
                        flattened = self.flatten_params(self.cached_experts[j].parameters())
                    req = dist.isend(flattened, dst=k)
                    send_requests.append(req)

            for req in send_requests:
                req.wait()  # Synchronize sends

    def request_expert_weights(self, expert_idx):
        with torch.no_grad():
            rank = expert_idx // self.num_experts_per_gpu

            flattened = torch.zeros(self.total_param_size, device='cuda')
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
                tokens[expert_mask[expert_idx]] = self.cached_experts[idx](tokens[expert_mask[expert_idx]])
            else:
                self.request_expert_weights(expert_idx)
                tokens[expert_mask[expert_idx]] = self.buffer_expert(tokens[expert_mask[expert_idx]])
            



    ###################### DANTE'S INFERNO ###########################

   # def __init__(self):
        # self.fixed_cache = fixed_cache
        # self.reference_cache = list(map(lambda x: x[:cache_size], reference_cache)) if reference_cache else None 
        # self.cache_policy = cache_policy


    #def validate_arguments():
                # if self.fixed_cache is not None:
        #     assert len(self.fixed_cache[0]) <= self.cache_size, "When specifying fixed_cache need to ensure cache_size is big enough."

        # match self.cache_policy:
        #     case "RAND": 
        #         self.gen_cache = self.gen_random_cache
        #     case "MRU":
        #         self.gen_cache = self.gen_most_recently_used_cache
        #     case "MTU":
        #         self.gen_cache = self.gen_most_tokens_used_cache
        #     case _:
        #         self.gen_cache = self.gen_same_cache


   # def cuda():
         # self.cache = [[None for _ in range(self.cache_size)] for _ in range(self.num_gpus)]
        # if self.fixed_cache:
        #     new_cache = self.fixed_cache[:]
        # elif self.reference_cache:
        #     new_cache = self.reference_cache[:]
        # else:
        #     new_cache = self.gen_random_cache(None) # Just gen randomly
        #new_cache[self.rank] = new_cache[self.rank].sort()

        # self.known_cache = new_cache
        # self.load_cache(new_cache)
        # self.expert_order = self.cache[self.rank].extend([x for x in range(self.num_experts) if x not in self.cache[self.rank]])



     # def load_cache(self, new_cache):
    #     new_rank_cache = new_cache[dist.get_rank()]
    #     new_rank_cache_aligned = [None for _ in range(self.cache_size)]

    #     loc = 0
    #     for expert in new_rank_cache:
    #         if expert in self.cache[dist.get_rank()]:
    #             # First need to check if there is something already there, if so need to move it
    #             idx = self.cache[dist.get_rank()].index(expert)
    #             if new_rank_cache_aligned[idx] != None:
    #                 orphan_expert = new_rank_cache_aligned[idx]
    #                 while new_rank_cache_aligned[loc] != None:
    #                     loc += 1
    #                 new_rank_cache_aligned[loc] = orphan_expert
    #                 loc += 1
    #             new_rank_cache_aligned[idx] = expert
    #         else:
    #             while new_rank_cache_aligned[loc] != None:
    #                 loc += 1
    #             new_rank_cache_aligned[loc] = expert
    #             loc += 1
        
    #     for slot_idx, expert_idx in enumerate(new_rank_cache_aligned):
    #         if expert_idx is not None:
    #             self.load_expert(expert_idx, slot_idx)

    #     for i in range(self.num_gpus):
    #         if i != dist.get_rank():
    #             self.cache[i] = new_cache[i]        
    
    # def get_cached(self):
    #     return copy.deepcopy(self.cache)
    
    
    # Returns index of expert.
    # def expert_loaded_location(self, expert_idx):
    #     # Returns -1 if not found
    #     idx = -1
    #     for i, v in enumerate(self.cache[dist.get_rank()]):
    #         if v == expert_idx:
    #             idx = i
    #             break
    #     # if idx == -1:
    #     #     raise Exception(f"Expert {expert_idx} is not cached on rank {dist.get_rank()}.")
    #     return idx 
    
    # def is_expert_loaded(self, expert_idx):
    #     return expert_idx in self.cache[dist.get_rank()]
    
    # def direct_execute_job(self, tokens, expert_mask, schedule=None):
    #     for expert_idx in range(self.num_experts):
    #         if expert_mask[expert_idx].shape[0] == 0:
    #             continue
    #         slot_idx = self.expert_loaded_location(expert_idx)
    #         tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])
        
    #     return tokens

    # def load_expert(self, expert_idx: int, load_idx: int, event = None): 
    #     if self.cache[self.rank][load_idx] == expert_idx:
    #         return
    #     self.cache[self.rank][load_idx] = expert_idx
    #     with torch.no_grad():
    #         if self.disable_async_fetch:
    #             for component in self.dynamic_components:
    #                 getattr(self.cached_experts[load_idx], component).weight.copy_(getattr(self.experts[expert_idx], component).weight)
    #             self.is_slot_loaded[load_idx].record()
    #         else:
    #             with torch.cuda.stream(self.load_stream):
    #                 if event is not None:
    #                     self.load_stream.wait_event(event)
    #                 with nvtx.annotate(f"Loading expert: {expert_idx} on slot idx {load_idx}", color="green"):
    #                     for component in self.dynamic_components:
    #                         getattr(self.cached_experts[load_idx], component).weight.copy_(getattr(self.experts[expert_idx], component).weight)
    #                 self.is_slot_loaded[load_idx].record()


    # def load_expert_into_slot(self, expert_idx, slot_idx):
    #     with torch.no_grad():
    #         with torch.cuda.stream(self.slot_streams[slot_idx]):
    #             for component in self.dynamic_components:
    #                 getattr(self.cached_experts[slot_idx], component).weight.copy_(getattr(self.experts[expert_idx], component).weight)



    # def execute_job(self, tokens, expert_mask, schedule):
    #     #expert_order = []
    #     #num_execs = 0
    #     #self.num_swaps = 0

    #     # if self.rank == 1:
    #     #     print(self.cache[self.rank])
    #     #     print(self.expert_order)
    #     # exit(0)

    #     expert_order = self.cache[self.rank][:]
    #     cached_experts_with_no_tokens = []
    #     for expert_idx in range(self.num_experts):
    #         if expert_mask[expert_idx].shape[0] == 0:
    #             if expert_idx in self.cache[self.rank]:
    #                 cached_experts_with_no_tokens.append(expert_idx)
    #         else:
    #             if expert_idx not in self.cache[self.rank]:
    #                 expert_order.append(expert_idx)
        
    #     for expert_idx in cached_experts_with_no_tokens:
    #         expert_order.remove(expert_idx)



    #     stale_slots = []
    #     for idx, expert_idx in enumerate(expert_order):
    #         # Check if expert has work to do
    #         # if expert_mask[expert_idx].shape[0] == 0:
    #         #     continue

    #         slot_idx = idx % self.cache_size # Which slot to execute the expert in

    #         with torch.cuda.stream(self.slot_streams[slot_idx]):
    #             # Fetch the expert except for the very first which will already be loaded
    #             if self.cache[self.rank][slot_idx] != expert_idx:
    #                 stale_slots.append(slot_idx)
    #                 self.load_expert_into_slot(expert_idx, slot_idx)

    #             # Perform computation 
    #             tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])


    #     # Maybe we want to synchronise before loading back in?
    #     # torch.cuda.synchronize(this device)
        
    #     # Load back every expert that was kicked out
    #     for stale_slot in stale_slots:
    #         self.load_expert_into_slot(self.cache[self.rank][stale_slot], stale_slot)
        
    #     return tokens      


    
    # def execute_job(self, tokens, expert_mask, schedule = None):
    #     expert_order = []
    #     num_execs = 0
    #     self.num_swaps = 0

    #     # Setup
    #     for expert_idx in range(self.num_experts):
    #         if expert_mask[expert_idx].shape[0] == 0:
    #             continue

    #         num_execs += 1
    #         if self.is_expert_loaded(expert_idx):
    #             expert_order.insert(0, expert_idx)
    #             # UNCESSESARY AS WE RELOAD CACHE EACH TIME self.is_slot_loaded[self.expert_loaded_location(expert_idx)].record()
    #         else:
    #             expert_order.append(expert_idx)

    #     unused_expert_slots = []
    #     # Start loading as many experts as possible
    #     for slot_idx, expert_idx in enumerate(self.cache[self.rank]):
    #         if expert_idx is None:
    #             next_expert_idx = self.next_not_loaded_expert(expert_order)
    #             if next_expert_idx is None:
    #                 break
    #             self.load_expert(next_expert_idx, slot_idx)
    #         elif expert_idx not in expert_order:
    #             unused_expert_slots.append(slot_idx)
       
    #     # Fill every unused expert
    #     for slot_idx in unused_expert_slots:
    #         next_expert_idx = self.next_not_loaded_expert(expert_order)
    #         if next_expert_idx == None:
    #             break
    #         self.load_expert(next_expert_idx, slot_idx)
        
        
    #     # Begin execution
    #     for idx, expert_idx in enumerate(expert_order):
    #         slot_idx = self.expert_loaded_location(expert_idx)
    #         # Check if not assigned to a cache which is possible if we have disabled async fetch
    #         if slot_idx == -1:
    #             # print("Need to fetch expert later")
    #             slot_idx = 0 # Just default to loading to the first slot
    #             self.num_swaps += 1
    #             self.load_expert(expert_idx, slot_idx) 

    #         # self.is_slot_loaded[slot_idx].synchronize()  # Wait until the expert is loaded
    #         torch.cuda.current_stream().wait_event(self.is_slot_loaded[slot_idx])
    #         #with torch.cuda.stream(self.comp_stream):
    #         #    self.comp_stream.wait_event(self.is_slot_loaded[slot_idx])
    #         with nvtx.annotate(f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"):
    #             tokens[expert_mask[expert_idx]] = self.cached_experts[slot_idx](tokens[expert_mask[expert_idx]])
    #         self.slot_finished_executing_events[slot_idx].record()

    #         # Check if anything else needs loading
    #         if not self.disable_async_fetch:
    #             next_expert_idx = self.next_not_loaded_expert(expert_order, idx)
    #             if next_expert_idx is not None:
    #                 self.num_swaps += 1
    #                 self.load_expert(next_expert_idx, slot_idx, event=self.slot_finished_executing_events[slot_idx])
        
    #     self.update_cache(schedule)

    #     self.num_swaps_iters.append(self.num_swaps)
        
    #     return tokens

        
    # def next_not_loaded_expert(self, experts, start=0):
    #     for idx in range(start,len(experts)):
    #         if experts[idx] not in self.cache[dist.get_rank()]:
    #             return experts[idx]
    #     return None

    # def update_cache(self, schedule):
    #     if schedule == None:
    #         return # Do not update cache 
        
    #     if self.fixed_cache:
    #         return # Do not update a fixed cache
        
    #     new_cache = self.gen_cache(schedule)

    #     if self.reference_cache:
    #         temp = self.reference_cache[:]
    #         for expert_idx in new_cache[dist.get_rank()]:
    #             if len(temp[dist.get_rank()]) >= self.cache_size:
    #                 break
    #             if expert_idx not in temp[dist.get_rank()]:
    #                 temp[dist.get_rank()].append(expert_idx)
    #         new_cache = temp
        
    #     self.known_cache = new_cache
    #     self.load_cache(new_cache)
    
    # def gen_same_cache(self, _):
    #     return self.cache
    
    # def gen_random_cache(self, _):
    #     expert_idxs = list(range(self.num_experts))
    #     self.rng.shuffle(expert_idxs)
    #     new_cache = [expert_idxs[i:i+self.cache_size] for i in range(0, self.cache_size*self.num_gpus, self.cache_size)]
    #     return new_cache
    
    # def gen_most_tokens_used_cache(self, schedule):
    #     new_cache = [[] for _ in range(self.num_gpus)]

    #     for i in range(self.num_gpus):
    #         expert_tokens = [0 for _ in range(self.num_experts)]
    #         for j in range(self.num_experts):
    #             for k in range(self.num_gpus):
    #                 expert_tokens[j] += schedule[k][j][i]

    #         experts_most_toks = map(lambda x: (x, expert_tokens[x]), range(self.num_experts))
    #         experts_most_toks = filter(lambda x: x[1] != 0, experts_most_toks)
    #         experts_most_toks = list(sorted(experts_most_toks, key=lambda x: x[1]))
    #         first_cache_size_experts_most_toks = list(map(lambda x: x[0], experts_most_toks[:self.cache_size]))
            
    #         new_cache[i] = first_cache_size_experts_most_toks
        
    #     return new_cache

    # def gen_most_recently_used_cache(self, schedule):
    #     # For this need to generate the work order
    #     pass 