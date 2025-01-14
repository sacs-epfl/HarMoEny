import torch
import nvtx


class AsynchronousCPU:
    def __init__(self, config):
        self.config = config
        self.local_cache = self.config.cache[self.config.rank]

        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.slot_ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.cache_size)
        ]
        self.expert_loaded = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.num_experts)
        ]
        self.finished_event = torch.cuda.Event(enable_timing=False)

        self.loaded_not_loaded = self.loaded_not_loaded_mixed

    def load_expert_into_slot(self, expert_idx, slot_idx):
        with nvtx.annotate(
            f"Loading expert {expert_idx} into slot {slot_idx}", color="green"
        ):
            with torch.no_grad():
                with torch.cuda.stream(self.load_stream):
                    self.load_stream.wait_event(self.slot_ready_events[slot_idx])

                    pinned_state_dict = self.config.experts[expert_idx]

                    cached_expert = self.config.cached_experts[slot_idx]
                    for name, param in cached_expert.named_parameters():
                        cpu_param = pinned_state_dict[name]
                        param.copy_(cpu_param, non_blocking=True)

                    self.expert_loaded[expert_idx].record(stream=self.load_stream)

    def loaded_not_loaded_ordered(self, expert_mask):
        loaded = []
        not_loaded = []

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

        return loaded, not_loaded

    def loaded_not_loaded_mixed(self, expert_mask):
        loaded = []
        not_loaded = []

        for i in range(self.config.num_experts):
            if expert_mask[i].shape[0] != 0:
                try:
                    slot_idx = self.local_cache.index(i)

                    # expert_idx, slot_idx, num_toks
                    loaded.append((i, slot_idx, expert_mask[i].shape[0]))
                except ValueError:
                    not_loaded.append(i)

        return loaded, not_loaded

    def generate_work_order(self, expert_mask):
        slots = []
        loaded, not_loaded = self.loaded_not_loaded(expert_mask)

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
        last_expert_exec = expert_order[-1] if len(expert_order) > 0 else None

        next_load_idx = 0
        for idx, expert_idx in enumerate(expert_order):
            slot_idx = slot_idxs[idx]

            with nvtx.annotate(
                f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"
            ):
                with torch.cuda.stream(self.comp_stream):
                    if expert_idx != self.local_cache[slot_idx]:
                        self.comp_stream.wait_event(self.expert_loaded[expert_idx])

                    # Execute the expert on the tokens
                    tokens[expert_mask[expert_idx]] = self.config.cached_experts[
                        slot_idx
                    ](tokens[expert_mask[expert_idx]])

                    self.slot_ready_events[slot_idx].record(stream=self.comp_stream)

            if next_load_idx < len(need_loading):  # Will need to load
                self.load_expert_into_slot(need_loading[next_load_idx], slot_idx)
                next_load_idx += 1
            elif (
                expert_idx != self.local_cache[slot_idx]
            ):  # Nothing else to load, do I need to bring in the original?
                self.load_expert_into_slot(self.local_cache[slot_idx], slot_idx)

            if expert_idx == last_expert_exec:
                if expert_idx != self.local_cache[slot_idx]:
                    self.finished_event.record(stream=self.load_stream)
                else:
                    self.finished_event.record(stream=self.comp_stream)

        self.finished_event.synchronize()
        return tokens
