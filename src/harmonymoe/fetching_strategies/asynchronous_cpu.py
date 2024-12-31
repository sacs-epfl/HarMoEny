import torch
import nvtx


class AsynchronousCPU:
    def __init__(self, config):
        self.config = config

        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.expert_finished_executing_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.num_experts)
        ]

        self.events = []

    def load_expert_into_slot(self, expert_idx, slot_idx):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with nvtx.annotate(
            f"Loading expert {expert_idx} into slot {slot_idx}", color="green"
        ):
            with torch.no_grad():
                with torch.cuda.stream(self.load_stream):
                    start_event.record(stream=self.load_stream)
                    pinned_state_dict = self.config.experts[expert_idx].state_dict()
                    cached_expert = self.config.cached_experts[slot_idx]
                    for name, param in cached_expert.named_parameters():
                        cpu_param = pinned_state_dict[name]
                        param.copy_(cpu_param, non_blocking=True)
                    end_event.record(stream=self.load_stream)

                    self.config.expert_loaded_events[expert_idx].record(
                        stream=self.load_stream
                    )

                    self.events.append((start_event, end_event, "LOAD"))

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

    def execute_job(self, tokens, expert_mask, schedule=None):
        expert_order, need_loading, slot_idxs = self.generate_work_order(expert_mask)
        
        global_start_event = torch.cuda.Event(enable_timing=True)
        global_start_event.record()
        self.events = []
        if self.config.rank == 1:
            print("-" * 45)

        for idx, expert_idx in enumerate(expert_order):
            slot_idx = slot_idxs[idx]

            if expert_idx != self.config.cache[self.config.rank][slot_idx]:
                self.comp_stream.wait_event(
                    self.config.expert_loaded_events[expert_idx]
                )

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with nvtx.annotate(
                f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"
            ):
                with torch.cuda.stream(self.comp_stream):
                    start_event.record(stream=self.comp_stream)
                    # Execute the expert on the tokens
                    tokens[expert_mask[expert_idx]] = self.config.cached_experts[
                        slot_idx
                    ](tokens[expert_mask[expert_idx]])
                    end_event.record(stream=self.comp_stream)

                    self.events.append((start_event, end_event, "EXECUTE"))

            # Record that the expert has finished executing
            self.expert_finished_executing_events[expert_idx].record(
                stream=self.comp_stream
            )

            if len(need_loading) != 0:  # Will need to load
                self.load_stream.wait_event(
                    self.expert_finished_executing_events[expert_idx]
                )
                self.load_expert_into_slot(need_loading.pop(0), slot_idx)
            elif (
                expert_idx != self.config.cache[self.config.rank][slot_idx]
            ):  # Nothing else to load, do I need to bring in the original?
                self.load_stream.wait_event(
                    self.expert_finished_executing_events[expert_idx]
                )
                self.load_expert_into_slot(
                    self.config.cache[self.config.rank][slot_idx], slot_idx
                )

        # PRINTINGS TEMPORARY
        for start_event, end_event, name_event in self.events:
            if self.config.rank == 1:
                relative_start_time = global_start_event.elapsed_time(start_event)
                end_event.synchronize()
                relative_end_time = global_start_event.elapsed_time(end_event)
                print(f"[{name_event}] Start: {relative_start_time} ms, End: {relative_end_time} ms")

        return tokens
