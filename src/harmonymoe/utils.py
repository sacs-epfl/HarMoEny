import torch.nn as nn
from copy import deepcopy 

from .moe_layer import MoELayer

def replace_moe_layer(model, moe_parent_type, moe_type, router_name, shared_experts, config):
    _replace_moe_layer(model, moe_parent_type, moe_type, [0], router_name, shared_experts, config)

    # return get_moe_layers([], model, target)

def _replace_moe_layer(model, moe_parent_type, moe_type, layer_idx, router_name, shared_experts, config):
    if type(model).__name__ == moe_parent_type:
        for child_name, child in model.named_children():
            if type(child).__name__ == moe_type:
                router = getattr(child, router_name)
                config.layer_idx = layer_idx[0]
                layer_idx[0] += 1
                new_moe_layer = MoELayer(router, shared_experts[config.layer_idx], config)

                setattr(model, child_name, new_moe_layer)
    else:
        for child in model.children():
            _replace_moe_layer(
                child,
                moe_parent_type,
                moe_type,
                layer_idx,
                router_name,
                shared_experts,
                config,
            )


    # for name, module in model.named_children():
    #     # Need to do this to update the MoE reference
    #     for child_name, child in module.named_children():
    #         if type(child).__name__ == target:
    #             router = getattr(child, router_name)
    #             config.layer_idx = layer_idx[0]
    #             layer_idx[0] += 1
    #             new_moe_layer = MoELayer(router, shared_experts, config)

    #             setattr(module, child_name, new_moe_layer)
    #         else:
    #             _replace_moe_layer(
    #                 child, 
    #                 target, 
    #                 is_decoder,
    #                 layer_idx,
    #                 router_name, 
    #                 experts_name,
    #                 decoder_name,
    #                 config,
    #             )  

def get_moe_layers(model):
    return _get_moe_layers([], model)

def _get_moe_layers(acc, model):
    for module in model.children():
        if isinstance(module, MoELayer):
            acc.append(module)
        else:
            acc = _get_moe_layers(acc, module)
    return acc


def get_moe_experts(model, moe_type, experts_name):
    return _get_moe_experts(nn.ModuleList(), model, moe_type, experts_name)

def _get_moe_experts(acc, model, moe_type, experts_name):
    if type(model).__name__ == moe_type:
        experts = getattr(model, experts_name)
        if isinstance(experts, nn.ModuleDict):
            experts = nn.ModuleList(experts.values())
        acc.append(deepcopy(experts))
    else:
        for module in model.children():
            acc = _get_moe_experts(acc, module, moe_type, experts_name)
    return acc