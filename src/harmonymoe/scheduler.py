import torch
import torch.nn as nn
import random
import json

import torch.distributed as dist

class Scheduler():
    def __init__(self, scheduling_policy="deepspeed", num_experts=8, eq_tokens=150, num_gpus=1, d_model=768, expert_placement=None, layer_idx=0):
        self.num_experts = num_experts
        self.eq_tokens = eq_tokens
        self.d_model = d_model
        self.num_gpus = num_gpus

        self.scheduling_policy = scheduling_policy

        self.is_fixed_placement = False
        self.is_reference_placement = False
        self.expert_to_gpu = None
        self.gpu_to_experts_list = None 

        match scheduling_policy:
            case "deepspeed":
                self.scheduler = self.schedule_fixed
                self.is_fixed_placement = True
                self.expert_to_gpu = self.generate_naive_expert_gpu_tensor()
            case "harmony":
                self.scheduler = self.schedule_harmony
                self.is_reference_placement = True
                self.expert_to_gpu = self.generate_naive_expert_gpu_tensor()
            case "even_split":
                self.scheduler = self.schedule_even_split
            case "drop":
                self.scheduler = self.schedule_drop
                self.is_fixed_placement = True
                self.expert_to_gpu = self.generate_naive_expert_gpu_tensor()
            case "exflow":
                self.scheduler = self.schedule_fixed
                self.is_fixed_placement = True
                if expert_placement == None:
                    print("EXFLOW CHOSEN BUT NO PLACEMENT CREATED")
                with open(expert_placement, "r") as f:
                    expert_placement = json.load(f)
                    if len(expert_placement) <= layer_idx:
                        self.expert_to_gpu = self.generate_naive_expert_gpu_tensor()
                    else:
                        # TODO need to fix the load system to work with this new approach
                        self.expert_to_gpu = torch.tensor(expert_placement[layer_idx], dtype=torch.uint8)
            case _:
                print("SCHEDULING POLICY NOT IMPLEMENTED")
                exit(1)

        if self.expert_to_gpu is not None:
            self.gpu_to_experts_list = self.generate_expert_gpu_to_gpu_expert(self.expert_to_gpu)  
    
    def prepare(self):
        self.rank = dist.get_rank()
        self.expert_to_gpu.cuda()
    
    def get_fixed_assign(self):
        if self.is_fixed_placement:
            return self.gpu_to_experts_list
        return None 
        #return self.fixed_assign
    
    def get_reference_assign(self):
        if self.is_reference_placement:
            return self.gpu_to_experts_list
        return None
        #return self.reference_assign 
    
    def __call__(self, meta):
        return self.scheduler(meta)
    
    def distribute_tokens(self, schedule: [[[int]]], hidden_states, router_mask, num_toks_send):
        tokens = torch.empty((num_toks_send, self.d_model), device="cuda")
        tokens_idx = 0
        send_splits = []
        rank = dist.get_rank()

        amount_expert_filled = [0] * self.num_experts

        for i in range(self.num_gpus):
            pre = tokens_idx
            for j in range(self.num_experts):
                amount = schedule[rank][j][i]
                if amount != 0:
                    start = amount_expert_filled[j]
                    tokens[tokens_idx:tokens_idx+amount] = hidden_states[router_mask[:,:,j]][start:start+amount]
                    tokens_idx += amount
                    amount_expert_filled[j] += amount
            post = tokens_idx
            send_splits.append(post-pre)
        
        return tokens, send_splits
    
    def gather_tokens(self, schedule: [[[int]]], tokens: torch.Tensor, hidden_states, router_mask):
        tokens_idx = 0
        experts_idx = [0 for _ in range(self.num_experts)]
        rank = dist.get_rank()

        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                amount = schedule[rank][j][i]
                if amount != 0:
                    start = experts_idx[j]
                    hidden_states[router_mask[:,:,j]][start:start+amount] = tokens[tokens_idx:tokens_idx+amount]
                    tokens_idx += amount
                    experts_idx[j] += amount

        return hidden_states 


    def allocate_recv_tensors(self, schedule: [[[int]]]):
        recv_splits = []
        rank = dist.get_rank()

        for i in range(self.num_gpus):
            num_tokens = 0
            for j in range(self.num_experts):
                num_tokens += schedule[i][j][rank]
            recv_splits.append(num_tokens)

        return torch.empty((sum(recv_splits), self.d_model), device="cuda"), recv_splits

    # def allocate_recv_tensors(self, schedule: [[[int]]]):
    #     recv_splits = torch.tensor(
    #         [sum(schedule[i][j][self.rank] for j in range(self.num_experts)) for i in range(self.num_gpus)],
    #         device="cuda"
    #     )

    #     return torch.empty((recv_splits.sum(), self.d_model), device="cuda"), recv_splits.tolist()


    def group_experts(self, schedule: [[[int]]], tokens: torch.Tensor):
        rank = dist.get_rank()

        num_toks_per_expert = [0 for _ in range(self.num_experts)]
        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                num_toks_per_expert[j] += schedule[i][j][rank]

        expert_tokens = [torch.empty((num_toks_per_expert[j], self.d_model), device="cuda") for j in range(self.num_experts)]
        experts_idx = [0 for _ in range(self.num_experts)]

        tokens_idx = 0
        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                amount = schedule[i][j][rank]
                if amount != 0:
                    start = experts_idx[j]
                    expert_tokens[j][start:start+amount] = tokens[tokens_idx:tokens_idx+amount]
                    tokens_idx += amount
                    experts_idx[j] += amount
        
        return expert_tokens
    
    def ungroup_experts(self, schedule: [[[int]]], expert_tokens: torch.Tensor, num_tokens):
        tokens = torch.empty((num_tokens, self.d_model), device="cuda")
        tokens_idx = 0
        expert_tokens_idx = [0 for _ in range(self.num_experts)]
        rank = dist.get_rank()

        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                amount = schedule[i][j][rank]
                if amount != 0:
                    start = expert_tokens_idx[j]
                    tokens[tokens_idx:tokens_idx+amount] = expert_tokens[j][start:start+amount]
                    tokens_idx += amount
                    expert_tokens_idx[j] += amount
        
        return tokens

    def generate_naive_expert_gpu_tensor(self):
        num_experts_per_gpu = self.num_experts // self.num_gpus

        placement = torch.arange(self.num_experts, dtype=torch.long) // num_experts_per_gpu

        return placement
    
    def generate_expert_gpu_to_gpu_expert(self, expert_gpu):
        return [
            (expert_gpu == i).nonzero(as_tuple=False).flatten() for i in range(self.num_gpus)
        ]

    def schedule_drop(self, meta):
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus) 
        gpu_amt = [0 for _ in range(self.num_gpus)]
        for i in range(self.num_gpus): # source
            for j in range(self.num_experts): # expert
                num_tokens = meta[i][j].item()
                if num_tokens == 0:
                    continue # We do not have work for this one
                start_idx = 0
                for k in range(self.num_gpus): # dest
                    if j in self.deepspeed_assign[k]:
                        if gpu_amt[k] < avg:
                            num_tokens_send = min(num_tokens, avg - gpu_amt[k])
                            schedule[i][j][k] = num_tokens_send
                            gpu_amt[k] += num_tokens_send

        return schedule
    
    def schedule_fixed(self, meta):
        schedule = torch.zeros((self.num_gpus, self.num_experts, self.num_gpus), dtype=torch.int, device="cuda")

        schedule[:, torch.arange(self.num_experts, device="cuda"), self.expert_to_gpu] = meta

        return schedule

    def schedule_harmony(self, meta):
        schedule = self.schedule_fixed(meta)

        avg = meta.sum().item() // self.num_gpus

        gpu_amt = torch.sum(schedule, dim=(0, 1)) # Shape (num_gpus,)

        while (gpu_amt > avg).any():
            # Identify most overload GPU
            overloaded_idx = torch.argmax(gpu_amt) # Long

            # Find GPU sending most tokens to overloaded GPU
            send_tokens = torch.sum(schedule[:, :, overloaded_idx], dim=1) # Shape (num_gpus,)
            sender_idx = torch.argmax(send_tokens)

            # Find the expert sending the most tokens from the sender
            expert_tokens = schedule[sender_idx, :, overloaded_idx]
            expert_idx = torch.argmax(expert_tokens) # Long

            tokens_to_move = expert_tokens[expert_idx]

            if tokens_to_move < self.eq_tokens:
                break # Not enough tokens to move
            
            # Identify least loaded GPU
            least_idx = torch.argmin(gpu_amt) # Long

            if least_idx == overloaded_idx or gpu_amt[least_idx] + self.eq_tokens > avg:
                break

            tokens_send = torch.min(tokens_to_move, avg - gpu_amt[least_idx])
          
            schedule[sender_idx, expert_idx, overloaded_idx] -= tokens_send
            schedule[sender_idx, expert_idx, least_idx] += tokens_send
            gpu_amt[overloaded_idx] -= tokens_send
            gpu_amt[least_idx] += tokens_send

        return schedule.tolist()

    def schedule_even_split(self, meta):
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                even = meta[i][j].item() // self.num_gpus
                leftover = meta[i][j].item() % self.num_gpus
                for k in range(self.num_gpus):
                    amt = even
                    if leftover > 0:
                        leftover -= 1
                        amt += 1
                    schedule[i][j][k] = amt 
        
        return schedule