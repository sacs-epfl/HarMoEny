import torch
import torch.nn as nn
import torch.distributed as dist

from fmoe.gates.naive_gate import NaiveGate
from fmoe.layers import FMoE
from router import Router 
from utils import TimedModule

class ExpertWrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child
        
        def forward(self, x, _):
            return self.child(x)
    
class SwitchFMoEWrapper(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(-1, d_model)  # Shape: [batch_size * seq_len, d_model]
        output = self.child(x)  # FMoE expects input of shape [tokens, d_model]
        output = output.reshape(batch_size, seq_len, d_model)
        return output

# FOR TESTING PURPOSES
class NaiveRouterWrapper(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(d_model, num_expert, world_size, top_k=top_k, gate_bias=gate_bias)
    
    def forward(self, x):
        gate_top_k_idx, gate_score = super().forward(x)
        print(f"Top_k_idx: {gate_top_k_idx.shape}")
        print(f"Score: {gate_score.shape}")
        exit(0)

# Update to add FMoE to it
def add_fmoe_model_switch(module, idx, args):
    if type(module).__name__ == "SwitchTransformersLayerFF":
        for child_name, child in module.named_children():
            if type(child).__name__ == "SwitchTransformersSparseMLP":
                router = getattr(child, "router")
                experts = getattr(child, "experts")
                if type(experts) == nn.ModuleDict:
                    experts = list(experts.values())
                
                num_experts_per_gpu = args.num_experts // args.world_size

                experts = experts[dist.get_rank()*num_experts_per_gpu:(dist.get_rank()+1)*num_experts_per_gpu]

                class RouterWrapper(Router):
                    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
                        super().__init__(args.num_experts, skew=args.router_skew, num_expert_skew=args.router_num_experts_skew, enable_random=args.random_router_skew)
                        self.d_model = d_model
                    
                    def forward(self, x):
                        router_mask, router_probs, router_logits = super().forward(x)
                        gate_top_k_idx = torch.argmax(router_mask, dim=-1).unsqueeze(1)
                        gate_score = router_probs

                        return gate_top_k_idx, gate_score

                new = FMoE(
                    num_expert=num_experts_per_gpu,
                    d_model=768,
                    world_size=args.world_size,
                    top_k=1,
                    expert=[lambda _, e=e: ExpertWrapper(e) for e in experts],
                    gate=RouterWrapper if args.enable_router_skew else NaiveGate,
                )

                if not args.enable_router_skew:
                    with torch.no_grad():
                        new.gate.gate.weight.copy_(router.classifier.weight)
                
                setattr(module, child_name, TimedModule(SwitchFMoEWrapper(new), idx=idx[0]))
                idx[0] += 1
    else:
        for child in module.children():
            add_fmoe_model_switch(child, idx, args)




def add_fmoe_model_mixtral(module, idx, args):
    if type(module).__name__ == "MixtralDecoderLayer":
        for child_name, child in module.named_children():
            if type(child).__name__ == "MixtralSparseMoeBlock":
                router = getattr(child, "gate")
                experts = getattr(child, "experts")

                num_experts_per_gpu = args.num_experts // args.world_size

                experts = experts[dist.get_rank()*num_experts_per_gpu:(dist.get_rank()+1)*num_experts_per_gpu]

                class RouterWrapper(Router):
                    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
                        super().__init__(args.num_experts, skew=args.router_skew, num_expert_skew=args.router_num_experts_skew, enable_random=args.random_router_skew)
                        self.d_model = d_model
                    
                    def forward(self, x):
                        router_mask, router_probs, router_logits = super().forward(x)
                        gate_top_k_idx = torch.argmax(router_mask, dim=-1).unsqueeze(1)
                        gate_score = router_probs

                        return gate_top_k_idx, gate_score

                new = FMoE(
                    num_expert=num_experts_per_gpu,
                    d_model=768,
                    world_size=args.world_size,
                    top_k=1,
                    expert=[lambda _, e=e: ExpertWrapper(e) for e in experts],
                    gate=RouterWrapper if args.enable_router_skew else NaiveGate,
                )

                if not args.enable_router_skew:
                    with torch.no_grad():
                        new.gate.gate.weight.copy_(router.weight)
                
                setattr(module, child_name, TimedModule(new))
                idx[0] += 1
    else:
        for child in module.children():
            add_fmoe_model_mixtral(child, idx, args)

