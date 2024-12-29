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

    def load_expert_into_slot(self, expert_idx, slot_idx):
        with nvtx.annotate(
            f"Loading expert {expert_idx} into slot {slot_idx}", color="green"
        ):
            with torch.no_grad():
                with torch.cuda.stream(self.load_stream):
                    pinned_state_dict = self.config.experts[expert_idx].state_dict()
                    cached_expert = self.config.cached_experts[slot_idx]
                    for name, param in cached_expert.named_parameters():
                        cpu_param = pinned_state_dict[name]
                        param.copy_(cpu_param, non_blocking=True)
                    self.config.expert_loaded_events[expert_idx].record(
                        stream=self.load_stream
                    )

    # TODO instead of calling .cache figure out what is loaded already
    def execute_job(self, tokens, expert_mask, schedule=None):
        expert_order = self.config.cache[self.config.rank][:]

        # Setup
        for expert_idx in range(self.config.num_experts):
            if expert_mask[expert_idx].shape[0] != 0 and expert_idx not in expert_order:
                expert_order.append(expert_idx)

        # Rare instance will the cached expert have no tokens to process which will be just a no-op

        # Begin execution
        stale_slots = []
        for idx, expert_idx in enumerate(expert_order):
            slot_idx = idx % self.config.cache_size

            if expert_idx != self.config.cache[self.config.rank][slot_idx]:
                self.comp_stream.wait_event(
                    self.config.expert_loaded_events[expert_idx]
                )

            with nvtx.annotate(
                f"Executing expert {expert_idx} on slot {slot_idx}", color="blue"
            ):
                with torch.cuda.stream(self.comp_stream):
                    # Execute the expert on the tokens
                    tokens[expert_mask[expert_idx]] = self.config.cached_experts[
                        slot_idx
                    ](tokens[expert_mask[expert_idx]])
            # Record that the expert has finished executing
            self.expert_finished_executing_events[expert_idx].record(
                stream=self.comp_stream
            )

            # Check if anything else needs loading
            if idx + self.config.cache_size < len(expert_order):
                stale_slots.append(slot_idx)
                self.load_stream.wait_event(
                    self.expert_finished_executing_events[expert_idx]
                )
                self.load_expert_into_slot(
                    expert_order[idx + self.config.cache_size], slot_idx
                )

        for stale_slot in stale_slots:
            self.load_expert_into_slot(
                self.config.cache[self.config.rank][stale_slot], stale_slot
            )

        return tokens
