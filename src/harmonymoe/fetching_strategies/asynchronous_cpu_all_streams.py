import torch
import nvtx
from dataclasses import dataclass
from concurrent.futures import wait, ThreadPoolExecutor

class AsynchronousCPUAllStreams:

    @dataclass
    class Command:
        name: str
        expert_idx: int
        slot_idx: int

    def __init__(self, config):
        self.config = config

    def generate_work_order(self, expert_mask):
        loaded = []
        not_loaded = []
        slots = []

        # First collect all experts that need to execute
        # Break them down between those loaded and not loaded
        for i in range(self.config.num_experts):
            if expert_mask[i].shape[0] != 0:
                if (
                    i >= self.config.first_slot_expert_idx
                    and i <= self.config.last_slot_expert_idx
                ):
                    # expert_idx, slot_idx, num_toks
                    loaded.append(
                        (
                            i,
                            i - self.config.first_slot_expert_idx,
                            expert_mask[i].shape[0],
                        )
                    )
                else:
                    not_loaded.append(i)

        loaded.sort(reverse=True, key=lambda x: x[2])
        loaded_slots = list(map(lambda x: x[1], loaded))
        loaded = list(map(lambda x: x[0], loaded))

        if len(loaded) == 0:
            return not_loaded, not_loaded, [0] * len(not_loaded)
        elif len(loaded) == 1:
            return (
                loaded + not_loaded,
                not_loaded,
                [loaded_slots[0]] * (1 + len(not_loaded)),
            )
        else:
            not_loaded_slots = [loaded_slots[i % 2] for i in range(len(not_loaded))]

            return (
                loaded[:2] + not_loaded + loaded[2:],
                not_loaded,
                loaded_slots[:2] + not_loaded_slots + loaded_slots[2:],
            )
    
    def group_execution_for_stream(self, work_order, not_loaded, slot_idxs):
        groups = [[] for _ in range(self.config.cache_size)]

        for i, expert_idx in enumerate(work_order):
            if expert_idx in not_loaded:
                groups[slot_idxs[i]].append(
                    self.Command(
                        name="LOAD",
                        expert_idx=expert_idx,
                        slot_idx=slot_idxs[i]
                    )
                )
            groups[slot_idxs[i]].append(
                self.Command(
                    name="EXECUTE",
                    expert_idx=expert_idx,
                    slot_idx=slot_idxs[i]
                )
            )            
        
        return groups


    def execute_job(self, tokens, expert_mask, schedule=None):
        work_order, need_loading, slot_idxs = self.generate_work_order(expert_mask)
        groups = self.group_execution_for_stream(work_order, need_loading, slot_idxs)
                           
        def start_execution(commands):
            with torch.no_grad(): # Move it in if want to support training
                with torch.cuda.stream(torch.cuda.Stream()):
                    for command in commands:
                        if command.name == "EXECUTE":
                            with nvtx.annotate(
                                f"Executing expert {command.expert_idx} on slot {command.slot_idx}", color="blue"
                            ):
                                tokens[expert_mask[command.expert_idx]] = self.config.cached_experts[
                                    command.slot_idx
                                ](tokens[expert_mask[command.expert_idx]])
                        elif command.name == "LOAD":
                            with nvtx.annotate(
                                f"Loading expert {command.expert_idx} into slot {command.slot_idx}", color="green"
                            ):
                                pinned_state_dict = self.config.experts[command.expert_idx].state_dict()
                                cached_expert = self.config.cached_experts[command.slot_idx]
                                for name, param in cached_expert.named_parameters():
                                    cpu_param = pinned_state_dict[name]
                                    param.copy_(cpu_param, non_blocking=True)
                        else:
                            raise Exception("NOT IMPLEMENTED")

        with ThreadPoolExecutor(max_workers=self.config.cache_size) as executor:
            futures = [executor.submit(start_execution, groups[i]) for i in range(self.config.cache_size)]
            for fut in futures:
                fut.result()
            
        return tokens