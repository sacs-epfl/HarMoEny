import torch
import torch.nn as nn
import random
import json


class Scheduler:
    def __init__(
        self,
        rank,
        scheduling_policy="deepspeed",
        num_experts=8,
        eq_tokens=150,
        num_gpus=1,
        d_model=768,
        expert_placement=None,
        layer_idx=0,
    ):
        self.num_experts = num_experts
        self.eq_tokens = eq_tokens
        self.d_model = d_model
        self.num_gpus = num_gpus

        self.scheduling_policy = scheduling_policy

        self.is_fixed_placement = False
        self.is_reference_placement = False
        self.expert_to_gpu = None
        self.gpu_to_experts_list = None

        self.rank = rank
        if self.expert_to_gpu is not None:
            self.expert_to_gpu.cuda()

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
                self.expert_to_gpu = self.generate_naive_expert_gpu_tensor()
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
                        self.gpu_to_experts_list = expert_placement[layer_idx]
                        self.expert_to_gpu = self.convert_gpu_to_experts_to_expert_gpu(
                            self.gpu_to_experts_list
                        )
            case _:
                print("SCHEDULING POLICY NOT IMPLEMENTED")
                exit(1)

        if self.expert_to_gpu is not None and self.gpu_to_experts_list is None:
            self.gpu_to_experts_list = self.generate_expert_gpu_to_gpu_expert(
                self.expert_to_gpu
            )

    def get_cache(self):
        if self.gpu_to_experts_list is not None:
            return self.gpu_to_experts_list
        else:
            return None

    def get_fixed_assign(self):
        if self.is_fixed_placement:
            return list(
                map(
                    lambda x: list(map(lambda y: y.item(), x)), self.gpu_to_experts_list
                )
            )
        return None

    def get_reference_assign(self):
        if self.is_reference_placement:
            return list(
                map(
                    lambda x: list(map(lambda y: y.item(), x)), self.gpu_to_experts_list
                )
            )
        return None

    def __call__(self, meta):
        return self.scheduler(meta)

    def distribute_tokens(self, schedule, hidden_states, expert_indices, num_toks_send):
        tokens = torch.empty(
            (num_toks_send, self.d_model), device="cuda", dtype=hidden_states.dtype
        )
        tokens_idx = 0
        send_splits = []

        amount_expert_filled = [0] * self.num_experts

        for i in range(self.num_gpus):
            pre = tokens_idx
            for j in range(self.num_experts):
                amount = schedule[self.rank][j][i]
                if amount != 0:
                    start = amount_expert_filled[j]
                    tokens[tokens_idx : tokens_idx + amount] = hidden_states[
                        expert_indices[j]
                    ][start : start + amount]
                    tokens_idx += amount
                    amount_expert_filled[j] += amount
            post = tokens_idx
            send_splits.append(post - pre)

        return tokens, send_splits

    def generate_distribution_mask(self, schedule, expert_indices, num_toks_send):
        indices = torch.empty(
            (num_toks_send,), dtype=torch.long, device=schedule.device
        )

        local_schedule = schedule[self.rank]

        send_splits = local_schedule.sum(dim=0)  # Shape (num_gpus,)
        tokens_gpu_offset = send_splits.cumsum(0) - send_splits  # Shape (num_gpus,)

        for j in range(self.num_experts):
            amounts = local_schedule[j]  # Shape (num_gpus,)
            non_zero_mask = amounts > 0

            if non_zero_mask.any():
                start = tokens_gpu_offset[non_zero_mask]
                end = start + amounts[non_zero_mask]
                tokens_gpu_offset[non_zero_mask] = end

                indices_index = torch.cat(
                    [
                        torch.arange(s, e, device=schedule.device)
                        for s, e in zip(start, end)
                    ]
                )

                indices[indices_index] = expert_indices[
                    j
                ]  # Assuming that all tokens are assigned. Can also do expert_amounts[j] instead perhaps if it isn't more expensive

        return indices, send_splits.tolist()

    def gather_tokens(
        self, schedule, tokens: torch.Tensor, hidden_states, expert_indices
    ):
        tokens_idx = 0
        experts_idx = [0 for _ in range(self.num_experts)]

        for i in range(self.num_gpus):
            for j in range(self.num_experts):
                amount = schedule[self.rank][j][i]
                if amount != 0:
                    start = experts_idx[j]
                    hidden_states[expert_indices[j]][start : start + amount] = tokens[
                        tokens_idx : tokens_idx + amount
                    ]
                    tokens_idx += amount
                    experts_idx[j] += amount

        return hidden_states

    def allocate_recv_tensors(self, schedule, dtype=torch.float):
        recv_splits = torch.sum(schedule[:, :, self.rank], dim=(1))  # Shape (num_gpus,)

        return (
            torch.empty((recv_splits.sum(), self.d_model), device="cuda", dtype=dtype),
            recv_splits.tolist(),
        )

    def generate_expert_mask(self, schedule: torch.Tensor, total_tokens):
        amounts = schedule[:, :, self.rank]  # Shape (num_gpus, num_experts)

        if total_tokens == 0:
            return [
                torch.tensor([], dtype=torch.long, device=amounts.device)
                for _ in range(self.num_experts)
            ]

        amounts_flat = amounts.view(-1)  # Shape (num_gpus * num_experts)
        expert_indices = torch.arange(self.num_experts, device=amounts.device).repeat(
            self.num_gpus
        )
        token_expert = torch.repeat_interleave(expert_indices, amounts_flat)
        token_indices = torch.arange(total_tokens, device=amounts.device)

        sorted_expert_values, sorted_indices = torch.sort(token_expert)

        # Get the sorted token indices
        sorted_token_indices = token_indices[sorted_indices]

        num_tokens_per_expert = amounts.sum(dim=0)  # Shape (num_experts,)

        # Split the sorted token indices according to the number of tokens per expert
        expert_indices_list = torch.split(
            sorted_token_indices, num_tokens_per_expert.tolist()
        )

        return expert_indices_list

    def generate_naive_expert_gpu_tensor(self):
        num_experts_per_gpu = self.num_experts // self.num_gpus

        placement = (
            torch.arange(self.num_experts, dtype=torch.long) // num_experts_per_gpu
        )

        return placement

    def generate_expert_gpu_to_gpu_expert(self, expert_gpu):
        return [
            (expert_gpu == i).nonzero(as_tuple=False).flatten().tolist()
            for i in range(self.num_gpus)
        ]

    def convert_gpu_to_experts_to_expert_gpu(self, gpu_experts):
        expert_gpu = torch.zeros(self.num_experts, dtype=torch.long)

        for i in range(self.num_gpus):
            for e in gpu_experts[i]:
                expert_gpu[e] = i

        return expert_gpu

    def schedule_drop(self, meta):
        schedule = self.schedule_fixed(meta)

        avg = meta.sum().item() // self.num_gpus

        gpu_amt = torch.sum(schedule, dim=(0, 1))

        for i in range(self.num_gpus):
            while gpu_amt[i] > avg:
                # Find entry that is sending the most

                over = gpu_amt[i] - avg

                schedule_for_gpu = schedule[:, :, i]

                max_val, max_idx = torch.max(schedule_for_gpu.view(-1), dim=0)
                if max_val == 0:
                    break  # This shouldn't happen given while loop

                idx_source = max_idx // self.num_experts
                idx_expert = max_idx % self.num_experts

                remove_amt = min(over, max_val.item())

                schedule[idx_source, idx_expert, i] -= remove_amt
                gpu_amt[i] -= remove_amt

        return schedule

    def schedule_fixed(self, meta):
        schedule = torch.zeros(
            (self.num_gpus, self.num_experts, self.num_gpus),
            dtype=torch.int,
            device="cuda",
        )

        schedule[
            :, torch.arange(self.num_experts, device="cuda"), self.expert_to_gpu
        ] = meta

        return schedule

    # OLD VERSION (used to emphasis timebreakdown)
    # def schedule_harmony(self, meta):
    #     schedule = self.schedule_fixed(meta)

    #     avg = meta.sum().item() // self.num_gpus

    #     gpu_amt = torch.sum(schedule, dim=(0, 1))  # Shape (num_gpus,)

    #     while (gpu_amt > avg).any():
    #         # Identify most overload GPU
    #         overloaded_idx = torch.argmax(gpu_amt)  # Long

    #         # Find GPU sending most tokens to overloaded GPU
    #         send_tokens = torch.sum(
    #             schedule[:, :, overloaded_idx], dim=1
    #         )  # Shape (num_gpus,)
    #         sender_idx = torch.argmax(send_tokens)

    #         # Find the expert sending the most tokens from the sender
    #         expert_tokens = schedule[sender_idx, :, overloaded_idx]
    #         expert_idx = torch.argmax(expert_tokens)  # Long

    #         tokens_to_move = expert_tokens[expert_idx]

    #         if tokens_to_move < self.eq_tokens:
    #             break  # Not enough tokens to move

    #         # Identify least loaded GPU
    #         least_idx = torch.argmin(gpu_amt)  # Long

    #         if least_idx == overloaded_idx or gpu_amt[least_idx] + self.eq_tokens > avg:
    #             break

    #         tokens_send = torch.min(tokens_to_move, avg - gpu_amt[least_idx])

    #         schedule[sender_idx, expert_idx, overloaded_idx] -= tokens_send
    #         schedule[sender_idx, expert_idx, least_idx] += tokens_send
    #         gpu_amt[overloaded_idx] -= tokens_send
    #         gpu_amt[least_idx] += tokens_send

    #     return schedule

    # Newer version which benefits from less global tokens
    def schedule_harmony(self, meta):
        schedule = self.schedule_fixed(meta)

        avg = meta.sum().item() // self.num_gpus

        gpu_amt = torch.sum(schedule, dim=(0, 1))  # Shape (num_gpus,)

        while (gpu_amt > avg).any():
            # Identify most overload GPU
            overloaded_idx = torch.argmax(gpu_amt)  # Long

            # Identify least loaded GPU
            least_idx = torch.argmin(gpu_amt)  # Long

            if least_idx == overloaded_idx or gpu_amt[least_idx] + self.eq_tokens > avg:
                break

            # Find most popular expert on overloaded_idx GPU
            send_tokens = torch.sum(
                schedule[:, :, overloaded_idx], dim=0
            )  # Shape (num_experts,)
            expert_idx = torch.argmax(send_tokens)

            if send_tokens[expert_idx] < self.eq_tokens:
                break  # Expert does not have enough summed senders to amortize

            amount_overloaded = gpu_amt[overloaded_idx] - avg
            senders = schedule[:, expert_idx, overloaded_idx]
            # Are we still overloaded? AND does our offload GPU still have room?
            while amount_overloaded > 0 and avg - gpu_amt[least_idx] > 0:
                sender_idx = torch.argmax(senders)
                tokens_to_move = senders[sender_idx]

                if tokens_to_move <= 0:
                    break  # We may have removed self.eq_tokens which is < than the amount overloaded

                tokens_send = torch.min(tokens_to_move, avg - gpu_amt[least_idx])

                schedule[
                    sender_idx, expert_idx, overloaded_idx
                ] -= tokens_send  # This will impact senders
                schedule[sender_idx, expert_idx, least_idx] += tokens_send

                gpu_amt[overloaded_idx] -= tokens_send
                gpu_amt[least_idx] += tokens_send

                amount_overloaded -= tokens_send

        return schedule

    def schedule_even_split(self, meta):
        schedule = [
            [[0 for _ in range(self.num_gpus)] for _ in range(self.num_experts)]
            for _ in range(self.num_gpus)
        ]

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

        return torch.tensor(schedule, dtype=torch.long, device="cuda")
