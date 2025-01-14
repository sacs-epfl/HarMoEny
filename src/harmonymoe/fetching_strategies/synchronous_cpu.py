import torch
import nvtx


class SynchronousCPU:
    def __init__(self, config):
        self.config = config
        self.load_finished = torch.cuda.Event(enable_timing=False)
        self.end_event = torch.cuda.Event(enable_timing=False)

    def load_expert_into_slot_synchronously(self, expert_idx, slot_idx):
        with nvtx.annotate(
            f"Loading expert {expert_idx} into slot {slot_idx}", color="green"
        ):
            with torch.no_grad():
                pinned_state_dict = self.config.experts[expert_idx]  # .state_dict()

                cached_expert = self.config.cached_experts[slot_idx]
                for name, param in cached_expert.named_parameters():
                    cpu_param = pinned_state_dict[name]
                    param.copy_(cpu_param, non_blocking=False)

                self.load_finished.record()
                self.load_finished.synchronize()

    def execute_job(self, tokens, expert_mask, schedule=None):
        expert_order = self.config.cache[self.config.rank][:]
        cached_experts_with_no_tokens = []
        for expert_idx in range(self.config.num_experts):
            if expert_mask[expert_idx].shape[0] == 0:
                if expert_idx in self.config.cache[self.config.rank]:
                    cached_experts_with_no_tokens.append(expert_idx)
            else:
                if expert_idx not in self.config.cache[self.config.rank]:
                    expert_order.append(expert_idx)

        for expert_idx in cached_experts_with_no_tokens:
            expert_order.remove(expert_idx)

        loaded = False

        for idx, expert_idx in enumerate(expert_order):
            if expert_idx not in self.config.cache[self.config.rank]:
                loaded = True
                slot_idx = 0
                self.load_expert_into_slot_synchronously(expert_idx, slot_idx)
            else:
                slot_idx = self.config.cache[self.config.rank].index(expert_idx)

            tokens[expert_mask[expert_idx]] = self.config.cached_experts[slot_idx](
                tokens[expert_mask[expert_idx]]
            )

        if loaded:
            self.load_expert_into_slot_synchronously(
                self.config.cache[self.config.rank][0], 0
            )
        self.end_event.record()
        self.end_event.synchronize()

        return tokens
