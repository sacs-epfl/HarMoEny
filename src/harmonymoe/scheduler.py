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

        self.fixed_assign = None
        self.reference_assign = None

        match scheduling_policy:
            case "deepspeed":
                self.scheduler = self.schedule_fixed
                self.fixed_assign = self.generate_deepspeed_topo()
            case "harmony":
                self.scheduler = self.schedule_harmony
                self.reference_assign = self.generate_deepspeed_topo()
            case "even_split":
                self.scheduler = self.schedule_even_split
            case "drop":
                self.scheduler = self.schedule_drop
                self.fixed_assign = self.generate_deepspeed_topo()
            case "adfabricus":
                self.scheduler = self.schedule_adfabricus
            case "exflow":
                self.scheduler = self.schedule_fixed
                if expert_placement == None:
                    print("EXFLOW CHOSEN BUT NO PLACEMENT CREATED")
                with open(expert_placement, "r") as f:
                    expert_placement = json.load(f)
                    if len(expert_placement) <= layer_idx:
                        self.fixed_assign = self.generate_deepspeed_topo()
                    else:
                        self.fixed_assign = expert_placement[layer_idx]
            case _:
                print("SCHEDULING POLICY NOT IMPLEMENTED")
                exit(1)
    
    def get_fixed_assign(self):
        return self.fixed_assign
    
    def get_reference_assign(self):
        return self.reference_assign 
    
    def __call__(self, meta, topo):
        return self.scheduler(meta, topo)
    
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



    # def distribute_tokens(self, schedule: [[[int]]], hidden_states, router_mask):
    #     distribution = [[] for _ in range(self.num_gpus)]

    #     for j in range(self.num_experts):
    #         start = 0
    #         for i in range(self.num_gpus):
    #             if schedule[dist.get_rank()][j][i] != 0:
    #                 size = schedule[dist.get_rank()][j][i]
    #                 distribution[i].append(hidden_states[router_mask[:,:,j]][start:start+size,:])
    #                 start += size

    #     return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in distribution]
    
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


    # Put appropriate updates into hidden_states
    # def gather_tokens(self, schedule: [[[int]]], gpu_tokens: [torch.Tensor], hidden_states, router_mask):
    #     expert_start_idx = [0 for _ in range(self.num_experts)]
    #     for i in range(self.num_gpus):
    #         start = 0
    #         for j in range(self.num_experts):
    #             if schedule[dist.get_rank()][j][i] != 0:
    #                 size = schedule[dist.get_rank()][j][i]
    #                 hidden_states[router_mask[:,:,j]][expert_start_idx[j]:expert_start_idx[j]+size,:] = gpu_tokens[i][start:start+size,:]
    #                 start += size

    #     return hidden_states

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
    #     recv = []

    #     for i in range(self.num_gpus):
    #         num_tokens = 0
    #         for j in range(self.num_experts):
    #             num_tokens += schedule[i][j][dist.get_rank()]
    #         recv.append(torch.empty((num_tokens, self.d_model), device="cuda"))
        
    #     return recv

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
    
    # A transformation to go from tokens index by GPU to expert 
    # def group_experts(self, schedule: [[[int]]], gpu_tokens: [torch.Tensor]):
    #     expert_tokens = [[] for _ in range(self.num_experts)]

    #     for i in range(self.num_gpus):
    #         start = 0
    #         end = 0
    #         for j in range(self.num_experts):
    #             if schedule[i][j][dist.get_rank()] != 0:
    #                 end += schedule[i][j][dist.get_rank()]
    #                 expert_tokens[j].append(gpu_tokens[i][start:end])
    #                 start = end

    #     return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in expert_tokens]
    

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


    # A transformation to go from tokens index by expert to GPU
    # def ungroup_experts(self, schedule: [[[int]]], expert_tokens: [torch.Tensor]):
    #     gpu_tokens = [[] for _ in range(self.num_gpus)]

    #     for j in range(self.num_experts):
    #         start = 0
    #         end = 0
    #         for i in range(self.num_gpus):
    #             if schedule[i][j][dist.get_rank()] != 0:
    #                 end += schedule[i][j][dist.get_rank()]
    #                 gpu_tokens[i].append(expert_tokens[j][start:end])
    #                 start = end

    #     return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in gpu_tokens]

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
    
    def schedule_fixed(self, meta, _):
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        expert_placement = self.fixed_assign if self.fixed_assign is not None else self.reference_assign

        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                target = -1
                for k in range(self.num_gpus):
                    if j in expert_placement[k]:
                        target = k
                        break
                if target == -1:
                    raise Exception("Failure on the deck: there is an expert overboard")
                schedule[i][j][target] = meta[i][j].item()
        return schedule

    def schedule_drop(self, meta, _):
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

    def schedule_harmony(self, meta, topology):
        schedule = self.schedule_fixed(meta, topology)
        avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus) 

        # Now we should rebalance it
        gpu_amt = [0 for _ in range(self.num_gpus)]

        # Let us first get all the amounts on each gpu 
        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                for k in range(self.num_gpus):
                    gpu_amt[i] += schedule[k][j][i]

        for i in range(self.num_gpus):
            while gpu_amt[i] > avg: # GPU i is received too many tokens
                # Find gpu sending me the most tokens
                most_tokens = 0
                sender_idx = -1
                for k in range(self.num_gpus):
                    tot = 0
                    for j in range(self.num_experts):
                        tot += schedule[k][j][i]
                    if tot > most_tokens:
                        most_tokens = tot
                        sender_idx = k
                
                if most_tokens == 0:
                    break # No one is sending tokens... just in case
                
                # Next find the expert sending the most from sender_idx
                most_tokens = 0
                expert_idx = -1
                for j in range(self.num_experts):
                    if schedule[sender_idx][j][i] > most_tokens:
                        most_tokens = schedule[sender_idx][j][i]
                        expert_idx = j
                
                if most_tokens == 0:
                    break # This should not happen as the previous check passed
                
                if most_tokens < self.eq_tokens:
                    break # Cannot do move, not enough tokens

                # Find GPU with least tokens
                least = gpu_amt[0]
                least_idx = 0
                for k in range(self.num_gpus):
                    if gpu_amt[k] < least:
                        least = gpu_amt[k]
                        least_idx = k
                
                if least < self.eq_tokens:
                    most_tokens -= self.eq_tokens
                    # If the offload does not have enough tokens we will need to 
                    # leave enough tokens on the overloaded to fetch the expert from system memory 
                
                if least == i:
                    break # Don't offload to yourself
                
                # Check if least has enough room
                if least + self.eq_tokens > avg:
                    break
                

                # Next we do the move now
                tokens_send = min(most_tokens, avg - least)
                schedule[sender_idx][expert_idx][i] -= tokens_send
                gpu_amt[i] -= tokens_send
                schedule[sender_idx][expert_idx][least_idx] += tokens_send
                gpu_amt[least_idx] += tokens_send
        
        return schedule

    def schedule_even_split(self, meta, _):
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
    
    def schedule_adfabricus(self, meta, topology):
        # num_gpus x num_experts x num_gpus
        # sender x expert x destination
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        gpu_amt = [0 for _ in range(self.num_gpus)]
        
        avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus)

        skipped = []

        # Try to place as according to topology much without breaking apart
        for gpu_idx in range(self.num_gpus):
            for expert_idx in range(self.num_experts):
                num_tokens = meta[gpu_idx][expert_idx].item()
                if num_tokens == 0:
                    continue
                found = False
                for offload_gpu_idx in range(self.num_gpus):
                    if expert_idx in topology[offload_gpu_idx]:
                        if num_tokens + gpu_amt[offload_gpu_idx] <= avg:
                            schedule[gpu_idx][expert_idx][offload_gpu_idx] += num_tokens
                            gpu_amt[offload_gpu_idx] += num_tokens
                            found = True
                            break
                if not found:
                    skipped.append([gpu_idx, expert_idx, num_tokens])
                
        # Now try to saturate the topology
        for i in range(len(skipped)):
            for offload_gpu_idx in range(self.num_gpus):
                if skipped[i][1] in topology[offload_gpu_idx]:
                    if gpu_amt[offload_gpu_idx] < avg: # Expert already exists so no need for eq tokens
                        tokens_send = min(skipped[i][2], avg - gpu_amt[offload_gpu_idx])
                        schedule[skipped[i][0]][skipped[i][1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        skipped[i][2] -= tokens_send
        
        # Remove from skipped any that has no more tokens
        temp = []
        for skip in skipped:
            if skip[2] != 0:
                temp.append(skip)
        skipped = temp

        gpu_experts = topology

        # Now will need to split across
        for skip in skipped:
            num_tokens = skip[2]
            # First look for any gpu with already the expert less than avg
            # This is not redundant to the above because a previous skip could have added a new expert to a GPU
            resolved = False
            for offload_gpu_idx in range(self.num_gpus):
                if num_tokens == 0:
                    break
                if skip[1] in gpu_experts[offload_gpu_idx]:
                    if gpu_amt[offload_gpu_idx] < avg:
                        tokens_send = min(num_tokens, avg - gpu_amt[offload_gpu_idx])
                        schedule[skip[0]][skip[1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        num_tokens -= tokens_send
            if num_tokens == 0:
                continue # Let us rebalance the next skipped token set

            # Then we will look for minimal GPU and if that GPU has atleast eq_tokens space less than avg
            while num_tokens > self.eq_tokens:
                min_gpu_idx = 0
                _min = gpu_amt[0]
                for offload_gpu_idx in range(self.num_gpus):
                    if gpu_amt[offload_gpu_idx] < _min:
                        _min = gpu_amt[offload_gpu_idx]
                        min_gpu_idx = offload_gpu_idx
                if _min + self.eq_tokens <= avg:
                    tokens_send = min(num_tokens, avg - _min)
                    schedule[skip[0]][skip[1]][min_gpu_idx] += tokens_send
                    gpu_experts[min_gpu_idx].append(skip[1])
                    gpu_amt[min_gpu_idx] += tokens_send
                    num_tokens -= tokens_send
                else:
                    break
            if num_tokens == 0:
                continue # Let us rebalance the next skipped token set

            # Otherwise we split evenly the tokens across the gpus with the expert already
            # Find all GPUs with the expert
            offloaders = []
            for offload_gpu_idx in range(self.num_gpus):
                if skip[1] in gpu_experts[offload_gpu_idx]:
                    offloaders.append(offload_gpu_idx)
            
            if len(offloaders) > 0:
                # Next we want to add tokens to each to make them balanced
                # This is not just evenly splitting since some GPUs may have more or less tokens
                target = int((sum(map(lambda x: gpu_amt[x], offloaders)) + num_tokens) / len(offloaders))
                # Fill each to the target
                for offload_gpu_idx in offloaders:
                    if num_tokens == 0:
                        break
                    if gpu_amt[offload_gpu_idx] < target:
                        tokens_send = min(num_tokens, target - gpu_amt[offload_gpu_idx])
                        schedule[skip[0]][skip[1]][offload_gpu_idx] += tokens_send
                        gpu_amt[offload_gpu_idx] += tokens_send
                        num_tokens -= tokens_send

                # If in really rare case there are still leftover tokens then just give all to the one with the least
                if num_tokens > 0:
                    min_offload_gpu_idx = offloaders[0]
                    _min = gpu_amt[offloaders[0]]
                    for offload_gpu_idx in offloaders:
                        if gpu_amt[offload_gpu_idx] < _min:
                            _min = gpu_amt[offload_gpu_idx]
                            min_offload_gpu_idx = offload_gpu_idx
                    schedule[skip[0]][skip[1]][min_offload_gpu_idx] += num_tokens
                    gpu_amt[min_offload_gpu_idx] += num_tokens
                    # We are guaranteed finished here
            else:
                # Rare: happens if there is no GPU with the expert
                # Just give it to the GPU with the least 
                min_offload_gpu_idx = 0
                _min = gpu_amt[0]
                for offload_gpu_idx in range(self.num_gpus):
                    if gpu_amt[offload_gpu_idx] < _min:
                        _min = gpu_amt[offload_gpu_idx]
                        min_offload_gpu_idx  = offload_gpu_idx
                schedule[skip[0]][skip[1]][min_offload_gpu_idx] += num_tokens
                gpu_amt[min_offload_gpu_idx] += num_tokens
                gpu_experts[min_offload_gpu_idx].append(skip[1])
                # We are guaranteed finished here

        return schedule