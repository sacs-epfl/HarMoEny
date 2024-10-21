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
        self.max_tokens_send_single_gpu = avg + self.eq_tokens * self.num_gpus


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
            # case "even_split":
            #     self.scheduler = self.schedule_even_split
            # case "drop":
            #     self.scheduler = self.schedule_drop
            #     self.fixed_assign = self.deepspeed_assign
            # case "demeter":
            #     self.scheduler = self.schedule_demeter
            #     self.reference_assign = self.deepspeed_assign
            # case "adfabricus":
            #     self.scheduler = self.schedule_adfabricus
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
            # if self.rank == 0:
            #     print(f"Expert {j} has: {hidden_states[router_mask[:,:,j]].shape} tokens")
            expert_toks = hidden_states[router_mask[:,:,j]]
            start = 0
            for i in range(self.num_gpus):
                if schedule[i][j] != 0:
                    size = schedule[i][j]
                    end = start + size 
                    end_gpu_idx = gpu_start_idx[i] + size
                    # if self.rank == 0:
                    #     print(f"Allocating to expert {j}: {end-start}")
                    distribution[i][gpu_start_idx[i]:end_gpu_idx] = expert_toks[start:end]
                    distribution[i][0][j] = size
                    start = end
                    gpu_start_idx[i] = end_gpu_idx
        
        #exit(0)
        return distribution

        # distribution = [[] for _ in range(self.num_gpus)]

        # for j in range(self.num_experts):
        #     start = 0
        #     for i in range(self.num_gpus):
        #         if schedule[self.rank][j][i] != 0:
        #             size = schedule[self.rank][j][i]
        #             distribution[i].append(hidden_states[router_mask[:,:,j]][start:start+size,:])
        #             start += size

        # return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in distribution]
    
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
        
        
        # expert_start_idx = [1 for _ in range(self.num_experts)]
        # for i in range(self.num_gpus):
        #     start = 0
        #     for j in range(self.num_experts):
        #         if schedule[self.rank][j][i] != 0:
        #             size = schedule[self.rank][j][i]
        #             hidden_states[router_mask[:,:,j]][expert_start_idx[j]:expert_start_idx[j]+size,:] = gpu_tokens[i][start:start+size,:]
        #             start += size

        # return hidden_states

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


        #return [torch.empty((1+self.max_tokens_send_single_gpu, self.d_model), device="cuda") for _ in range(self.num_gpus)]
        # recv = []

        # for i in range(self.num_gpus):
        #     num_tokens = 0
        #     for j in range(self.num_experts):
        #         num_tokens += schedule[i][j][self.rank]
        #     recv.append(torch.empty((num_tokens, self.d_model), device="cuda"))
        
        # return recv
    
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


        # expert_tokens = [[] for _ in range(self.num_experts)]

        # for i in range(self.num_gpus):
        #     start = 0
        #     end = 0
        #     for j in range(self.num_experts):
        #         if schedule[i][j][self.rank] != 0:
        #             end += schedule[i][j][self.rank]
        #             expert_tokens[j].append(gpu_tokens[i][start:end])
        #             start = end

        # return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in expert_tokens]

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

                    # end_expert_idx = start_expert_idx[j] + size
                    # tokens[i][start:end] = expert_tokens[j][start_expert_idx[j]:end_expert_idx]
                    # start = end
                    # start_expert_idx[j] = end_expert_idx
        
        return tokens


        # gpu_tokens = [[] for _ in range(self.num_gpus)]

        # for j in range(self.num_experts):
        #     start = 0
        #     end = 0
        #     for i in range(self.num_gpus):
        #         if schedule[i][j][self.rank] != 0:
        #             end += schedule[i][j][self.rank]
        #             gpu_tokens[i].append(expert_tokens[j][start:end])
        #             start = end

        # return [torch.cat(tokens, dim=0) if len(tokens) != 0 else torch.empty((0, self.d_model), device="cuda") for tokens in gpu_tokens]
    
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


    # def schedule_drop(self, meta, _):
    #     schedule = [[[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)] for _ in range(self.num_gpus)]

    #     avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus) 
    #     gpu_amt = [0 for _ in range(self.num_gpus)]
    #     for i in range(self.num_gpus): # source
    #         for j in range(self.num_experts): # expert
    #             num_tokens = meta[i][j].item()
    #             if num_tokens == 0:
    #                 continue # We do not have work for this one
    #             start_idx = 0
    #             for k in range(self.num_gpus): # dest
    #                 if j in self.deepspeed_assign[k]:
    #                     if gpu_amt[k] < avg:
    #                         num_tokens_send = min(num_tokens, avg - gpu_amt[k])
    #                         schedule[i][j][k] = num_tokens_send
    #                         gpu_amt[k] += num_tokens_send

    #     return schedule

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

    # def schedule_adnexus(self, meta, topology):
    #     schedule = self.schedule_deepspeed(meta, topology)
    #     avg = int(sum(map(lambda t: sum(t).item(), meta)) / self.num_gpus) 

    #     # Now we should rebalance it
    #     gpu_amt = [0 for _ in range(self.num_gpus)]

    #     # Let us first get all the amounts on each gpu 
    #     for i in range(self.num_gpus):
    #         for j in range(self.num_experts):
    #             for k in range(self.num_gpus):
    #                 gpu_amt[i] += schedule[k][j][i]

    #     for i in range(self.num_gpus):
    #         while gpu_amt[i] > avg: # GPU i is received too many tokens
    #             # Find gpu sending me the most tokens
    #             most_tokens = 0
    #             sender_idx = -1
    #             for k in range(self.num_gpus):
    #                 tot = 0
    #                 for j in range(self.num_experts):
    #                     tot += schedule[k][j][i]
    #                 if tot > most_tokens:
    #                     most_tokens = tot
    #                     sender_idx = k
                
    #             if most_tokens == 0:
    #                 break # No one is sending tokens... just in case
                
    #             # Next find the expert sending the most from sender_idx
    #             most_tokens = 0
    #             expert_idx = -1
    #             for j in range(self.num_experts):
    #                 if schedule[sender_idx][j][i] > most_tokens:
    #                     most_tokens = schedule[sender_idx][j][i]
    #                     expert_idx = j
                
    #             if most_tokens == 0:
    #                 break # This should not happen as the previous check passed
                
    #             if most_tokens < self.eq_tokens:
    #                 break # Cannot do move, not enough tokens

    #             # Find GPU with least tokens
    #             least = gpu_amt[0]
    #             least_idx = 0
    #             for k in range(self.num_gpus):
    #                 if gpu_amt[k] < least:
    #                     least = gpu_amt[k]
    #                     least_idx = k
                
    #             if least == i:
    #                 break # Don't offload to yourself
                
    #             # Check if least has enough room
    #             if least + self.eq_tokens > avg:
    #                 break
                

    #             # Next we do the move now
    #             tokens_send = min(most_tokens, avg - least)
    #             schedule[sender_idx][expert_idx][i] -= tokens_send
    #             gpu_amt[i] -= tokens_send
    #             schedule[sender_idx][expert_idx][least_idx] += tokens_send
    #             gpu_amt[least_idx] += tokens_send
        
    #     return schedule

    
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
                for gpu_idx, offloads in enumerate(self.demeter_offload_assign):
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
        
        #print(schedule)
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