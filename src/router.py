import torch.nn as nn 
import torch

class Router(nn.Module):
    def __init__(self, num_experts, skew=0):
        super().__init__()
        self.num_experts = num_experts
        self.skew = skew
    
    def forward(self, x):
        prob = torch.full((self.num_experts,), 1.0 / self.num_experts, device=x.device)
        if self.skew > 0:
            prob[0] += self.skew
            prob = prob / prob.sum()

        if x.dim() == 3:
            expert_indices = torch.multinomial(prob, num_samples=x.shape[0] * x.shape[1], replacement=True)
            expert_indices = expert_indices.view(x.shape[0], x.shape[1])  # Reshape to match the batch size and sequence length

            one_hot_experts = torch.zeros(x.shape[0], x.shape[1], self.num_experts, dtype=torch.float, device=x.device)
            one_hot_experts.scatter_(2, expert_indices.unsqueeze(-1), 1)

            router_probabilities = torch.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)
        else: # dim should be 2
            expert_indices = torch.multinomial(prob, num_samples=x.shape[0], replacement=True)

            one_hot_experts = torch.zeros(x.shape[0], self.num_experts, dtype=torch.float, device=x.device)
            one_hot_experts.scatter_(1, expert_indices.unsqueeze(-1), 1)

            router_probabilities = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        return one_hot_experts, router_probabilities, one_hot_experts