import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class RouterConfig:
    d_model: int
    num_experts: int
    weights: any
    model_dtype: str = None
    enable_skew: bool = False
    enable_random: bool = False
    enable_uniform: bool = False
    skew: int = 0.05
    num_experts_skewed: int = 1


class Router(nn.Module):
    def __init__(self, config):
        super(Router, self).__init__()

        self.config = config

        if self.config.skew > 1:
            self.config.skew = self.config.skew / 100

        if config.enable_skew:
            self.forward_exec = self.skew_forward
        elif config.enable_random:
            self.forward_exec = self.random_forward
        elif config.enable_uniform:
            self.forward_exec = self.uniform_forward
        else:
            self.forward_exec = self.standard_forward
            self.router = nn.Linear(
                self.config.d_model,
                self.config.num_experts,
                bias=False,
                dtype=self.config.weights.dtype,
            )
            if self.config.weights is not None:
                with torch.no_grad():
                    self.router.weight.copy_(self.config.weights)

    def forward(self, x):
        return self.forward_exec(x)

    def standard_forward(self, x):
        logits = self.router(x)
        probs = nn.functional.softmax(logits, dim=-1)

        expert_indices = torch.argmax(probs, dim=-1)
        expert_indices = nn.functional.one_hot(
            expert_indices, num_classes=self.config.num_experts
        )

        probs = torch.max(probs, dim=-1).values.unsqueeze(-1)

        return expert_indices, probs, logits

    def skew_forward(self, x):
        # skewed_multinomial_prob = self.config.skew / self.config.num_experts_skewed
        # multinomial_prob = (1.0 - self.config.skew) / (
        #     self.config.num_experts - self.config.num_experts_skewed
        # )

        # multinomial_probs = torch.full(
        #     (self.config.num_experts,), multinomial_prob, device=x.device
        # )
        # multinomial_probs[: self.config.num_experts_skewed] = skewed_multinomial_prob

        non_skewed_proportion = 1 / (self.config.skew + self.config.num_experts)
        skewed_proportion = (self.config.num_experts_skewed + self.config.skew) / (self.config.num_experts_skewed * (self.config.skew + self.config.num_experts))

        multinomial_probs = torch.full(
            (self.config.num_experts,), non_skewed_proportion, device=x.device
        )
        multinomial_probs[:self.config.num_experts_skewed] = skewed_proportion
        # multinomial_probs[:self.config.num_experts_skewed] += self.config.skew / self.config.num_experts_skewed
        # multinomial_probs[self.config.num_experts_skewed:] -= self.config.skew / (self.config.num_experts - self.config.num_experts_skewed)

        num_tokens_per_expert = (multinomial_probs * x.shape[0]).floor().long()

        disrepancy = x.shape[0] - num_tokens_per_expert.sum()
        for i in range(disrepancy): # Disrepancy must be less than self.config.num_experts
            multinomial_probs[i] += 1
        
        expert_indices = []
        for expert_idx, num_tokens in enumerate(num_tokens_per_expert):
            expert_indices.append(torch.full((num_tokens.item(),), expert_idx, dtype=torch.long, device=x.device))
        
        expert_indices = torch.cat(expert_indices)

        # expert_indices = torch.multinomial(
        #     multinomial_probs, num_samples=x.shape[0], replacement=True
        # )

        expert_indices = nn.functional.one_hot(
            expert_indices, num_classes=self.config.num_experts
        )

        probs = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        return expert_indices, probs, expert_indices.to(dtype=torch.float)

    def uniform_forward(self, x):
        prob = torch.full(
            (self.config.num_experts,), 1.0 / self.config.num_experts, device=x.device
        )

        expert_indices = torch.multinomial(
            prob, num_samples=x.shape[0], replacement=True
        )

        one_hot_experts = torch.zeros(
            x.shape[0], self.config.num_experts, dtype=torch.float, device=x.device
        )
        one_hot_experts.scatter_(1, expert_indices.unsqueeze(-1), 1)

        probs = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        return one_hot_experts, probs, one_hot_experts

    def random_forward(self, x):
        expert_index = torch.randint(
            0, self.config.num_experts, (x.shape[0],), dtype=torch.long, device=x.device
        )
        expert_index = nn.functional.one_hot(
            expert_index, num_classes=self.config.num_experts
        )

        probs = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        return expert_index, probs, expert_index
