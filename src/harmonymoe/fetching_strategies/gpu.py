import torch
import nvtx
import torch.distributed as dist


class GPU:
    def __init__(self, config):
        self.config = config

        self.num_experts_per_gpu = self.config.num_experts // self.config.world_size
        self.work_order = self.generate_work_order()
        self.total_param_size = sum(
            p.numel() for p in self.config.buffer_expert.parameters()
        )

        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

    def generate_work_order(self):
        experts = [i for i in range(self.config.num_experts)]
        start = self.config.first_slot_expert_idx
        end = self.config.last_slot_expert_idx
        work_order = experts[start : end + 1] + experts[:start] + experts[end + 1 :]
        return work_order

    def get_expert_processing_ranks(self, schedule):
        return schedule[
            :,
            self.config.first_slot_expert_idx : self.config.last_slot_expert_idx + 1,
            :,
        ].sum(dim=0)

    def flatten_params(self, params):
        return torch.cat([p.view(-1).to(dtype=torch.float) for p in params], dim=0)

    def unflatten_params(self, flattened, params):
        offset = 0
        for p in params:
            size = p.numel()
            p.data.copy_(flattened[offset : offset + size].view_as(p))
            offset += size

    def send_overloaded_weights(self, schedule):
        expert_processing_ranks = self.get_expert_processing_ranks(schedule)

        with torch.cuda.stream(self.load_stream):
            send_requests = []
            for j in range(self.config.cache_size):
                flattened = None
                for k in range(self.config.world_size):
                    if k == self.config.rank or expert_processing_ranks[j][k] == 0:
                        continue
                    if flattened is None:
                        flattened = self.flatten_params(
                            self.config.cached_experts[j].parameters()
                        )
                    req = dist.isend(flattened, dst=k)
                    send_requests.append(req)

            for req in send_requests:
                req.wait()  # Synchronize sends

    def request_expert_weights(self, expert_idx):
        with torch.no_grad():
            rank = expert_idx // self.num_experts_per_gpu
            flattened = torch.zeros(
                self.total_param_size, dtype=torch.float, device="cuda"
            )
            req = dist.irecv(flattened, src=rank)
            req.wait()  # Synchronize receive

            self.unflatten_params(flattened, self.config.buffer_expert.parameters())

    def execute_job(self, tokens, expert_mask, schedule=None):
        # Issue to comm stream the expert sends
        self.send_overloaded_weights(schedule)

        # Perform work
        for idx, expert_idx in enumerate(self.work_order):
            # Do we have work to do?
            if expert_mask[expert_idx].shape[0] == 0:
                continue

            # Is it a cached expert?
            if idx < self.config.cache_size:
                tokens[expert_mask[expert_idx]] = self.config.cached_experts[idx](
                    tokens[expert_mask[expert_idx]]
                )
            else:
                self.request_expert_weights(expert_idx)
                tokens[expert_mask[expert_idx]] = self.config.buffer_expert(
                    tokens[expert_mask[expert_idx]]
                )
