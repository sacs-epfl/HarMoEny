import torch
import nvtx
from dataclasses import dataclass
from concurrent.futures import as_completed, ThreadPoolExecutor

class AsynchronousCPUAllStreams:

    @dataclass
    class Command:
        name: str
        expert_idx: int
        slot_idx: int

    def __init__(self, config):
        self.config = config

        #self.streams = [torch.cuda.Stream() for _ in range(self.config.cache_size)]

        #self.events = []

    def load_expert_into_slot(self, expert_idx, slot_idx):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # with nvtx.annotate(
        #     f"Loading expert {expert_idx} into slot {slot_idx}", color="green"
        # ):
        with torch.no_grad():
            with torch.cuda.stream(self.streams[slot_idx]):
                start_event.record(stream=self.streams[slot_idx])
                pinned_state_dict = self.config.experts[expert_idx].state_dict()
                cached_expert = self.config.cached_experts[slot_idx]
                for name, param in cached_expert.named_parameters():
                    cpu_param = pinned_state_dict[name]
                    param.copy_(cpu_param, non_blocking=True)
                end_event.record(stream=self.streams[slot_idx])

        self.events.append((start_event, end_event, f"LOAD expert {expert_idx} slot {slot_idx}"))

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
                           
        def start_execution(commands, stream):
            local_events = []

            with torch.no_grad(): # Move it in if want to support training
                with torch.cuda.stream(stream):
                    for command in commands:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        start_event.record()
                        if command.name == "EXECUTE":
                            tokens[expert_mask[command.expert_idx]] = self.config.cached_experts[
                                command.slot_idx
                            ](tokens[expert_mask[command.expert_idx]])
                        elif command.name == "LOAD":
                            pinned_state_dict = self.config.experts[command.expert_idx].state_dict()
                            cached_expert = self.config.cached_experts[command.slot_idx]
                            for name, param in cached_expert.named_parameters():
                                cpu_param = pinned_state_dict[name]
                                param.copy_(cpu_param, non_blocking=True)
                        else:
                            raise Exception("NOT IMPLEMENTED")
                        end_event.record()

                        local_events.append((start_event, end_event, f"{command.name} expert {command.expert_idx} slot {command.slot_idx}"))

            return local_events

        with ThreadPoolExecutor(max_workers=self.config.cache_size) as executor:
            global_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.config.cache_size)]
            streams = [torch.cuda.Stream() for _ in range(self.config.cache_size)]
            for i, event in enumerate(global_start_events):
                event.record(stream=streams[i])
            grouped_events = []

            futures = [executor.submit(start_execution, groups[i], streams[i]) for i in range(self.config.cache_size)]
            for fut in as_completed(futures):
                stream_events = fut.result()
                grouped_events.append(stream_events)

            if self.config.rank == 1:
                print("-" * 45)
                for i, events in enumerate(grouped_events):
                    for start_event, end_event, name_event in events:
                        end_event.synchronize()
                        relative_start_time = global_start_events[i].elapsed_time(start_event)
                        relative_end_time = global_start_events[i].elapsed_time(end_event)
                        print(f"[{name_event}] Start: {relative_start_time} ms, End: {relative_end_time} ms")
            

        # for idx, expert_idx in enumerate(expert_order):
        #     slot_idx = slot_idxs[idx]

        #     start_event = torch.cuda.Event(enable_timing=True)
        #     end_event = torch.cuda.Event(enable_timing=True)

        #     # with nvtx.annotate(
        #     #     f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"
        #     # ):
        #     with torch.cuda.stream(self.streams[slot_idx]):
        #         start_event.record(stream=self.streams[slot_idx])
        #         # Execute the expert on the tokens
        #         tokens[expert_mask[expert_idx]] = self.config.cached_experts[
        #             slot_idx
        #         ](tokens[expert_mask[expert_idx]])
        #         end_event.record(stream=self.streams[slot_idx])

        #     self.events.append((start_event, end_event, f"EXECUTE expert {expert_idx} slot {slot_idx}"))


        #     if len(need_loading) != 0:  # Will need to load
        #         self.load_expert_into_slot(need_loading.pop(0), slot_idx)
        #     elif (
        #         expert_idx != self.config.cache[self.config.rank][slot_idx]
        #     ):  # Nothing else to load, do I need to bring in the original?
        #         self.load_expert_into_slot(
        #             self.config.cache[self.config.rank][slot_idx], slot_idx
        #         )

        # PRINTINGS TEMPORARY

        # for start_event, end_event, name_event in events:
        #     if self.config.rank == 1:
        #         relative_start_time = global_start_event.elapsed_time(start_event)
        #         #end_event.synchronize()
        #         relative_end_time = global_start_event.elapsed_time(end_event)
        #         print(f"[{name_event}] Start: {relative_start_time} ms, End: {relative_end_time} ms")

        return tokens