import torch
import torch.nn as nn
import torch.distributed as dist

from transformers.activations import ACT2FN
from fmoe.layers import FMoE


class FMoETransformerMLP(FMoE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, inp: torch.Tensor):
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        reshaped_output = output.reshape(original_shape)
        return reshaped_output

# Need to modify the forward of the Switch class 
# since FastMoE decides to give an extra value to forward
class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states, _):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states



def replace_fmoe_layer(model, target, router_name, experts_name, decoder_name, config):
    _replace_fmoe_layer(model, target, [0], router_name, experts_name, decoder_name, config)

    return get_fmoe_layers([], model, target)

def _replace_fmoe_layer(model, target, layer_idx, router_name, experts_name, decoder_name, config):
    for name, module in model.named_children():
        for child_name, child in module.named_children():
            if type(child).__name__ == target:
                router = getattr(child, router_name)
                experts = getattr(child, experts_name)
                if isinstance(experts, nn.ModuleDict):
                    experts = nn.ModuleList(experts.values())


                num_experts = len(experts)
                world_size = dist.get_world_size()
                num_experts_per_worker = num_experts // world_size

                new_moe_layer = FMoETransformerMLP(
                    num_expert=num_experts_per_worker,
                    d_model=config.d_model,
                    top_k=1,
                    world_size=world_size,
                    expert=lambda _: SwitchTransformersDenseActDense(config),
                )

                # Move the weights over
                with torch.no_grad():
                    # Router
                    new_moe_layer.gate.gate.weight.copy_(router.classifier.weight)

                    # Experts
                    rank = dist.get_rank()
                    start_idx = rank * num_experts_per_worker
                    end_idx = start_idx + num_experts_per_worker
                    for i in range(start_idx, end_idx):
                        new_moe_layer.experts[i-start_idx].wi.weight.copy_(experts[i].wi.weight)
                        new_moe_layer.experts[i-start_idx].wo.weight.copy_(experts[i].wo.weight)

                layer_idx[0] += 1

                setattr(module, child_name, new_moe_layer)
            else:
                _replace_fmoe_layer(
                    child, 
                    target, 
                    layer_idx,
                    router_name, 
                    experts_name,
                    decoder_name,
                    config,
                )  

def get_fmoe_layers(acc, model, target):
    for module in model.children():
        if isinstance(module, FMoETransformerMLP):
            acc.append(module)
        else:
            acc = get_fmoe_layers(acc, module, target)
    return acc