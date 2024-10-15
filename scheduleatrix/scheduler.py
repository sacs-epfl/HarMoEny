import torch
import torch.nn as nn
import random

import torch.distributed as dist

class Scheduler():
    def __init__(self, scheduling_policy="deepspeed", num_experts=8, eq_tokens=150, num_gpus=None, d_model=768):
        if num_gpus is None:
            self.rank = dist.get_rank()
            self.num_gpus = dist.get_world_size()
        else: # We are running unit tests
            self.rank = 0
            self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.eq_tokens = eq_tokens
        self.d_model = d_model

        self.deepspeed_topo = self.generate_deepspeed_topo()
        self.demeter_offload_topo = self.generate_deepspeed_topo()
        random.shuffle(self.demeter_offload_topo)

        match scheduling_policy:
            case "deepspeed":
                self.scheduler = self.schedule_deepspeed
            case "adnexus":
                self.scheduler = self.schedule_adnexus
            case "even_split":
                self.scheduler = self.schedule_even_split
            case "drop":
                self.scheduler = self.schedule_drop
            case "demeter":
                self.scheduler = self.schedule_demeter
            case "adfabricus":
                self.scheduler = self.schedule_adfabricus
            case _:
                print("SCHEDULING POLICY NOT IMPLEMENTED")
                exit(1)
    
    def __call__(self, meta, topo):
        return self.scheduler(meta, topo)

    def distribute_tokens(self, schedule: [[[int]]], hidden_states, router_mask):
        distribution = [[] for _ in range(self.num_gpus)]

        for j in range(self.num_experts):
            start = 0
            for i in range(self.num_gpus):
                if schedule[self.rank][j][i] != 0:
                    size = schedule[self.rank][j][i]
                    distribution[i].append(hidden_states[router_mask[:,:,j]][start:start+size,:])
                    start += size

        return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in distribution]
    
    # Put appropriate updates into hidden_states
    def gather_tokens(self, schedule: [[[int]]], gpu_tokens: [torch.Tensor], hidden_states, router_mask):
        expert_start_idx = [0 for _ in range(self.num_experts)]
        for i in range(self.num_gpus):
            start = 0
            for j in range(self.num_experts):
                if schedule[self.rank][j][i] != 0:
                    size = schedule[self.rank][j][i]
                    hidden_states[router_mask[:,:,j]][expert_start_idx[j]:expert_start_idx[j]+size,:] = gpu_tokens[i][start:start+size,:]
                    start += size

        return hidden_states

    def allocate_recv_tensors(self, schedule: [[[int]]]):
        recv = []

        for i in range(self.num_gpus):
            num_tokens = 0
            for j in range(self.num_experts):
                num_tokens += schedule[i][j][self.rank]
            recv.append(torch.empty((num_tokens, self.d_model), device="cuda"))
        
        return recv
    
    # A transformation to go from tokens index by GPU to expert 
    def group_experts(self, schedule: [[[int]]], gpu_tokens: [torch.Tensor]):
        expert_tokens = [[] for _ in range(self.num_experts)]

        for i in range(self.num_gpus):
            start = 0
            end = 0
            for j in range(self.num_experts):
                if schedule[i][j][self.rank] != 0:
                    end += schedule[i][j][self.rank]
                    expert_tokens[j].append(gpu_tokens[i][start:end])
                    start = end

        return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in expert_tokens]

    # A transformation to go from tokens index by expert to GPU
    def ungroup_experts(self, schedule: [[[int]]], expert_tokens: [torch.Tensor]):
        gpu_tokens = [[] for _ in range(self.num_gpus)]

        for j in range(self.num_experts):
            start = 0
            end = 0
            for i in range(self.num_gpus):
                if schedule[i][j][self.rank] != 0:
                    end += schedule[i][j][self.rank]
                    gpu_tokens[i].append(expert_tokens[j][start:end])
                    start = end

        return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in gpu_tokens]
    
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
    
    def schedule_deepspeed(self, meta, _):
        schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

        for i in range(self.num_gpus): # source
            for j in range(self.num_experts): # expert
                target = -1
                for k in range(self.num_gpus):
                    if j in self.deepspeed_topo[k]:
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
                    if j in self.deepspeed_topo[k]:
                        if gpu_amt[k] < avg:
                            num_tokens_send = min(num_tokens, avg - gpu_amt[k])
                            schedule[i][j][k] = num_tokens_send
                            gpu_amt[k] += num_tokens_send

        return schedule

    def schedule_adnexus(self, meta, topology):
        schedule = self.schedule_deepspeed(meta, topology)
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

    
    def schedule_demeter(self, meta, topology):
        schedule = self.schedule_deepspeed(meta, topology)
        avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus)

        gpu_amt = [0 for _ in range(self.num_gpus)]

        # Let us first get all the amounts on each gpu 
        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                for k in range(self.num_gpus):
                    gpu_amt[i] += schedule[k][j][i]

        for i in range(self.num_gpus):
            while gpu_amt[i] > avg:
            #    if self.rank == 0:
              #      print(f"ITER: {i}")
                # Find expert with most
                most = -1
                most_idx = -1
                for j in range(self.num_experts):
                    tot = 0
                    for k in range(self.num_gpus):
                        tot += schedule[k][j][i]
                    if tot > most:
                        most = tot
                        most_idx = j 

                if most == 0:
                    break
                
               # if self.rank == 0:
               #     print(f"Most: {most_idx}")


                # Find gpu that is sending that expert the most
                sender_idx = -1
                sender_amt = -1
                for j in range(self.num_gpus):
                    if schedule[j][most_idx][i] > sender_amt:
                        sender_amt = schedule[j][most_idx][i]
                        sender_idx = j
                        break 
                
                if sender_idx == -1:
                    raise Exception("Impossible exception occured")

             #   if self.rank == 0:
               #     print(f"Sender: {sender_idx}")

                # Find offload GPU for that expert
                offload_gpu_idx = -1
                for gpu_idx, offloads in enumerate(self.demeter_offload_topo):
                    if most_idx in offloads:
                        offload_gpu_idx = gpu_idx
                        break
                if offload_gpu_idx == -1:
                    raise Exception("Buck or two have we crashed a galley of good oars.")

                if gpu_amt[offload_gpu_idx] >= avg:
                    break 

             #   if self.rank == 0:
               #     print(f"Offload: {offload_gpu_idx}")

                #print(f"GPU:{i} overloaded moving expert:{most_idx} from GPU:{sender_idx} to GPU:{offload_gpu_idx}")
                # Move as much as possible
                # If cannot move more than eq_tokens break 
                num_tokens = min(sender_amt, avg - gpu_amt[offload_gpu_idx])
             #   if self.rank == 0:
               #     print(num_tokens)
                schedule[sender_idx][most_idx][i] -= num_tokens
                schedule[sender_idx][most_idx][offload_gpu_idx] += num_tokens
                gpu_amt[i] -= num_tokens
                gpu_amt[offload_gpu_idx] += num_tokens
            #print(f"Done: {i}")
        
        print(schedule)
        return schedule


    
    # OUTDATED
    # Do not use as has not been updated to the latest engine requirements
    # def schedule_demeter(self, hidden_states, router_mask, topology):
    #     expert_inputs = [hidden_states[router_mask[:,:,idx]] for idx in range(self.num_experts)]
    #     expert_sizes = [expert_inputs[i].shape[0] for i in range(self.num_experts)]
    #     avg = sum(expert_sizes) // self.num_gpus
    #     multiplier = 1.15
    #     avg_multiplier = int(multiplier * avg)

    #     allocation = [[0 for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

    #     # Step 1: Initial Allocation
    #     for gpu_idx, experts in enumerate(topology):
    #         for expert in experts:
    #             allocation[gpu_idx][expert] = expert_sizes[expert]

    #     # Step 2: Rebalance
    #     for i in range(self.num_gpus):
    #         while sum(allocation[i]) > avg_multiplier:
    #             # Get largest expert
    #             max_expert = -1
    #             max_amount = -1
    #             for j in range(self.num_experts):
    #                 if allocation[i][j] > max_amount:
    #                     max_amount = allocation[i][j]
    #                     max_expert = j


    #             # Get the offload GPU for that expert
    #             offload_gpu = -1
    #             for j in range(self.num_gpus):
    #                 if j != i and max_expert in topology[(j+1)%self.num_gpus]:
    #                     offload_gpu = j
    #                     break
                
    #             # Calculate maximal amount that can be shared 
    #             amount_to_share = min(sum(allocation[i])-avg_multiplier, allocation[i][max_expert])
    #             amount_to_share = min(amount_to_share, avg-sum(allocation[offload_gpu]))

    #             # If maximal amount is zero then break
    #             if amount_to_share <= 0:
    #                 break

    #             # Update
    #             allocation[i][max_expert] -= amount_to_share
    #             allocation[offload_gpu][max_expert] += amount_to_share

        
    #     # Building schedule
    #     schedule = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

    #     for j in range(self.num_experts):
    #         start = 0
    #         end = 0
    #         for i in range(self.num_gpus):
    #             if allocation[i][j] == 0:
    #                 continue
    #             end += allocation[i][j]
    #             schedule[i][j] = (start,end,expert_inputs[j][start:end])
    #             start = end 

    #     return schedule

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