import torch.nn as nn
import torch
from copy import deepcopy
import dataclasses

from .moe_layer import MoELayer
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersTop1Router,
)
from transformers.models.switch_transformers.configuration_switch_transformers import (
    SwitchTransformersConfig,
)


def get_tensor_by_path(module, path):
    parts = path.split(".")
    current = module
    for part in parts:
        current = getattr(current, part)
    return current


def replace_moe_layer(
    model, moe_parent_type, moe_type, name_experts, router_tensor_path, config
):
    _replace_moe_layer(
        model,
        moe_parent_type,
        moe_type,
        [0],
        name_experts,
        router_tensor_path,
        config,
    )


def _replace_moe_layer(
    model,
    moe_parent_type,
    moe_type,
    layer_idx,
    name_experts,
    router_tensor_path,
    config,
):
    if type(model).__name__ == moe_parent_type:
        for child_name, child in model.named_children():
            if type(child).__name__ == moe_type:
                experts = get_tensor_by_path(child, name_experts)
                if isinstance(experts, nn.ModuleDict):
                    experts = list(experts.values())
                elif isinstance(experts, nn.ModuleList):
                    experts = list(experts)

                pinned_experts = []
                for expert in experts:
                    # Create a pinned state_dict for each expert
                    pinned_state_dict = {
                        name: param.detach().cpu().pin_memory()
                        for name, param in expert.state_dict().items()
                    }
                    pinned_experts.append(pinned_state_dict)

                local_config = dataclasses.replace(config)
                local_config.layer_idx = layer_idx[0]
                local_config.experts = pinned_experts
                local_config.expert_example = experts[0]
                router = get_tensor_by_path(
                    child, router_tensor_path
                )
                local_config.router_weights = router.weight
                new_moe_layer = MoELayer(local_config)

                layer_idx[0] += 1

                setattr(model, child_name, new_moe_layer)
    else:
        for child in model.children():
            _replace_moe_layer(
                child,
                moe_parent_type,
                moe_type,
                layer_idx,
                name_experts,
                router_tensor_path,
                config,
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
