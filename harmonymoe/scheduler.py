import torch
import torch.nn as nn
import random

import torch.distributed as dist

class Scheduler():
    def __init__(self, scheduling_policy="deepspeed", num_experts=8, eq_tokens=150, num_gpus=None, d_model=768, max_num_toks_local=12000):
        if num_gpus is None:
            self.rank = dist.get_rank()
            self.num_gpus = dist.get_world_size()
        else: # We are running unit tests
            self.rank = 0
            self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.eq_tokens = eq_tokens
        self.d_model = d_model
        self.num_experts_per_gpu = self.num_experts // self.num_gpus
        self.max_num_toks_local = max_num_toks_local
        avg = self.max_num_toks_local // self.num_gpus 
        self.max_tokens_send_single_gpu = avg + self.eq_tokens #* self.num_gpus #TODO refine


        self.deepspeed_assign = self.generate_deepspeed_topo()
        self.demeter_offload_assign = self.generate_deepspeed_topo()
        random.shuffle(self.demeter_offload_assign)

        self.fixed_assign = None
        self.reference_assign = None

        match scheduling_policy:
            case "deepspeed":
                self.scheduler = self.schedule_deepspeed
                self.fixed_assign = self.deepspeed_assign
            case "adnexus":
                self.scheduler = self.schedule_adnexus
                self.reference_assign = self.deepspeed_assign
            case _:
                print("SCHEDULING POLICY NOT IMPLEMENTED")
                exit(1)
    
    def get_fixed_assign(self):
        return self.fixed_assign
    
    def get_reference_assign(self):
        return self.reference_assign 
    
    def __call__(self, meta, topo):
        return self.scheduler(meta, topo)

    def distribute_tokens(self, schedule: [[[int]]], hidden_states, router_mask):
        distribution = self.allocate_recv_tensors()
        
        gpu_start_idx = [1 for _ in range(self.num_gpus)]
        for j in range(self.num_experts):
            expert_toks = hidden_states[router_mask[:,:,j]]
            start = 0
            for i in range(self.num_gpus):
                if schedule[i][j] != 0:
                    size = schedule[i][j]
                    end = start + size 
                    end_gpu_idx = gpu_start_idx[i] + size
                    distribution[i][gpu_start_idx[i]:end_gpu_idx] = expert_toks[start:end]
                    distribution[i][0][j] = size
                    start = end
                    gpu_start_idx[i] = end_gpu_idx
        
        return distribution


    # Put appropriate updates into hidden_states
    def gather_tokens(self, tokens: [torch.Tensor], hidden_states, router_mask):
        expert_start_idx = [0 for _ in range(self.num_experts)]
        for i in range(self.num_gpus):
            start = 0
            for j in range(self.num_experts):
                size = int(tokens[i][0][j].item())
                if size != 0:
                    end = start + size
                    expert_end_idx = expert_start_idx[j] + size 
                    hidden_states[router_mask[:,:,j]][expert_start_idx[j]:expert_end_idx] = tokens[i][start:end]
                    start = end
                    expert_start_idx[j] = expert_end_idx
        
        return hidden_states
        
    def allocate_recv_tensors(self):
        return [
            torch.cat(
                [
                    torch.zeros((1, self.d_model), device="cuda"),
                    torch.empty((self.max_tokens_send_single_gpu, self.d_model), device="cuda")
                ],
                dim=0,
            )
            for _ in range(self.num_gpus)
        ]

    # A transformation to go from tokens index by GPU to expert 
    def group_experts(self, tokens: [torch.Tensor]):
        expert_tokens = [[] for _ in range(self.num_experts)]

        for i in range(self.num_gpus):
            start = 1
            for j in range(self.num_experts):
                size = int(tokens[i][0][j].item())
                if size != 0:
                    end = start + size
                    expert_tokens[j].append(tokens[i][start:end])
                    start = end
        
        return [torch.cat(toks, dim=0) if len(toks) != 0 else torch.empty((0, self.d_model), device="cuda") for toks in expert_tokens]


    # A transformation to go from tokens index by expert to GPU
    def ungroup_experts(self, expert_tokens: [torch.Tensor], tokens: [torch.Tensor]):
        gpu_tokens = self.allocate_recv_tensors()

        gpu_start_idx = [1 for _ in range(self.num_experts)]
        for j in range(self.num_experts):
            start = 0
            for i in range(self.num_gpus):
                size = int(tokens[i][0][j].item())
                if size != 0: 
                    #print(size)
                    end = start + size
                    gpu_end_idx = gpu_start_idx[i] + size
                    tokens[i][gpu_start_idx[i]:gpu_end_idx] = expert_tokens[j][start:end]
                    start = end
                    gpu_start_idx[i] = gpu_end_idx
        
        return tokens

    def generate_deepspeed_topo(self):
        topology = [[] for _ in range(self.num_gpus)]
        num_experts_per_gpu = self.num_experts // self.num_gpus
        leftover = self.num_experts % self.num_gpus
        start = 0
        end = 0 # Not inclusive
        for i in range(self.num_gpus):
            end += num_experts_per_gpu
            if leftover > 0:
                end += 1
                leftover -= 1
            
            for j in range(start,end):
                topology[i].append(j)

            start = end
        return topology
    
    def schedule_deepspeed(self, expert_freq, _):
        schedule = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        gpu_tok_amt = [0 for _ in range(self.num_gpus)]

        # Start with the basic append across 
        for expert_idx in range(self.num_experts):
            for gpu_idx in range(self.num_gpus):
                if expert_idx in self.deepspeed_assign[gpu_idx]:
                    if expert_freq[expert_idx] != 0:
                        schedule[gpu_idx][expert_idx] = expert_freq[expert_idx]
                        gpu_tok_amt[gpu_idx] += expert_freq[expert_idx]
        
        return schedule 

    def schedule_adnexus(self, expert_freq, topology):
        schedule = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        gpu_tok_amt = [0 for _ in range(self.num_gpus)]

        # Start with the basic append across 
        for expert_idx in range(self.num_experts):
            for gpu_idx in range(self.num_gpus):
                if expert_idx in self.deepspeed_assign[gpu_idx]:
                    if expert_freq[expert_idx] != 0:
                        schedule[gpu_idx][expert_idx] = expert_freq[expert_idx]
                        gpu_tok_amt[gpu_idx] += expert_freq[expert_idx]

        avg = sum(gpu_tok_amt) // self.num_gpus

        for i in range(self.num_gpus):
            while gpu_tok_amt[i] > avg:
                # Find the expert getting the most tokens
                expert_idx_most = 0
                expert_amt = schedule[i][0]
                for j in range(1, self.num_experts):
                    if schedule[i][j] > expert_amt:
                        expert_amt = schedule[i][j]
                        expert_idx_most = j

                # Find the GPU receiving the least
                gpu_idx_least = 0
                gpu_amt = gpu_tok_amt[0]
                for k in range(1, self.num_gpus):
                    if gpu_tok_amt[k] < gpu_amt:
                        gpu_amt = gpu_tok_amt[k]
                        gpu_idx_least = k

                # If itself then break
                if gpu_idx_least == i:
                    break

                # If least does not have atleast eq_tokens room
                if gpu_amt + self.eq_tokens > avg:
                    break

                # Try to move as many tokens
                num_toks_move = min(
                    avg - gpu_amt, # How much room is left without going over
                    expert_amt # Maximum number of tokens this expert can share
                )

                # If less than eq_tokens then break 
                if num_toks_move < self.eq_tokens:
                    break

                # Update 
                schedule[i][expert_idx_most] -= num_toks_move
                schedule[gpu_idx_least][expert_idx_most] += num_toks_move 

                gpu_tok_amt[i] -= num_toks_move
                gpu_tok_amt[gpu_idx_least] += num_toks_move

        
        # print(list(map(lambda arr: sum(arr), schedule)))
        # print(f"Max sent to a single GPU: {self.max_tokens_send_single_gpu}")
        # exit(0)
        return schedule 