import torch.nn as nn
import torch
from copy import deepcopy 

from .moe_layer import MoELayer
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router
from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig

def replace_moe_layer(model, moe_parent_type, moe_type, router_name, shared_experts, config, override_router=None):
    _replace_moe_layer(model, moe_parent_type, moe_type, [0], router_name, shared_experts, config, override_router=override_router)

def _replace_moe_layer(model, moe_parent_type, moe_type, layer_idx, router_name, shared_experts, config, override_router=None):
    if type(model).__name__ == moe_parent_type:
        for child_name, child in model.named_children():
            if type(child).__name__ == moe_type:
                if override_router == None:
                    local_router = getattr(child, router_name)
                    router = local_router

                    if moe_type == "MixtralSparseMoeBlock":
                        router_config = SwitchTransformersConfig(
                            num_experts=config.num_experts,
                            hidden_size=config.d_model,
                        )
                        router = SwitchTransformersTop1Router(router_config)
                else:
                    router = override_router()

                config.layer_idx = layer_idx[0]
                layer_idx[0] += 1
                new_moe_layer = MoELayer(router, shared_experts[config.layer_idx], config)

                if override_router == None:
                    if moe_type == "MixtralSparseMoeBlock":
                        with torch.no_grad():
                            new_moe_layer.router.classifier.weight.copy_(local_router.weight)
                        print("Moved data for new router")

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
                override_router=override_router,
            )

def get_moe_layers(model):
    return _get_moe_layers([], model)

def _get_moe_layers(acc, model):
    for module in model.children():
        if isinstance(module, MoELayer):
            acc.append(module)
        else:
            acc = _get_moe_layers(acc, module)
    return acc


# def get_moe_experts(model, moe_type, experts_name):
#     return _get_moe_experts(nn.ModuleList(), model, moe_type, experts_name)

# def _get_moe_experts(acc, model, moe_type, experts_name):
#     if type(model).__name__ == moe_type:
#         experts = getattr(model, experts_name)
#         if isinstance(experts, nn.ModuleDict):
#             experts = nn.ModuleList(experts.values())
#         acc.append(deepcopy(experts))
#     else:
#         for module in model.children():
#             acc = _get_moe_experts(acc, module, moe_type, experts_name)
#     return acc

def get_moe_experts(model, moe_type, experts_name):
    return _get_moe_experts([], model, moe_type, experts_name)

def _get_moe_experts(acc, model, moe_type, experts_name):
    if type(model).__name__ == moe_type:
        experts = getattr(model, experts_name)
        if isinstance(experts, nn.ModuleDict):
            experts = list(experts.values())
        elif isinstance(experts, nn.ModuleList):
            experts = list(experts)
        acc.append(deepcopy(experts))
    else:
        for module in model.children():
            acc = _get_moe_experts(acc, module, moe_type, experts_name)
    return acc