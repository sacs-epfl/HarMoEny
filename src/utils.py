import csv

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersDenseActDense


############## Timed Dense ####################
class TimedDense(SwitchTransformersDenseActDense):
    def __init__(self, layer_idx, is_decoder, **kwargs):
        super().__init__(**kwargs)

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.latencies = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def forward(self, x: torch.Tensor):
        self.start_event.record()
        x = super().forward(x)
        self.end_event.record()
        self.end_event.synchronize()
        self.latencies.append(self.start_event.elapsed_time(self.end_event))
        return x
    
    def save_statistics(self, DIR):
        path = f"{DIR}/dense"
        if self.is_decoder:
            path += "_decoder"
        path += f"_layer-{self.layer_idx}"

        with open(f"{path}.csv", "w") as f:
            fieldnames = ["iteration", "latency (ms)"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.latencies)):
                dic = {
                    "iteration": i,
                    "latency (ms)": self.latencies[i],
                }
             
                writer.writerow(dic)

def replace_dense_layer(model, target, decoder_name, config):
    _replace_dense_layer(model, target, [0], False, decoder_name, config)

    return get_dense_layers([], model, target)

def _replace_dense_layer(model, target, layer_idx, is_decoder, decoder_name, config):
    for name, module in model.named_children():
        if name == decoder_name:
            is_decoder = True 
            layer_idx[0] = 0
        elif type(module).__name__ == target:
            mlp = getattr(module, "mlp")
            if type(mlp).__name__ == "SwitchTransformersDenseActDense":
                new_dense = TimedDense(layer_idx[0], is_decoder, config=config)

                # Move the weights over
                with torch.no_grad():
                    new_dense.wi.weight.copy_(mlp.wi.weight)
                    new_dense.wo.weight.copy_(mlp.wo.weight)

                layer_idx[0] += 1

                setattr(module, "mlp", new_dense)
        else:
            _replace_dense_layer(
                module, 
                target, 
                layer_idx,
                is_decoder,
                decoder_name,
                config,
            )  

def get_dense_layers(acc, model, target):
    for module in model.children():
        if isinstance(module, TimedDense):
            acc.append(module)
        else:
            acc = get_dense_layers(acc, module, target)
    return acc
    


############## FastMoE ########################
from fmoe.layers import FMoE

class FMoETransformerMLP(FMoE):
    def __init__(self, layer_idx=0, is_decoder=False, **kwargs):
        super().__init__(**kwargs)

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.latencies = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def forward(self, inp: torch.Tensor):
        self.start_event.record()
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        reshaped_output = output.reshape(original_shape)
        self.end_event.record()
        self.end_event.synchronize()
        self.latencies.append(self.start_event.elapsed_time(self.end_event))
        return reshaped_output
    
    def save_statistics(self, DIR=""):
        path = f"{DIR}/moe"
        if self.is_decoder:
            path += "_decoder"
        path += f"_layer-{self.layer_idx}"

        with open(f"{path}.csv", "w") as f:
            fieldnames = ["iteration", "latency (ms)"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.latencies)):
                dic = {
                    "iteration": i,
                    "latency (ms)": self.latencies[i],
                }
             
                writer.writerow(dic)

# Need to modify the forward of the Switch class 
# since FastMoE decides to give an extra value to forward
class FMoESwitchTransformersDenseActDense(nn.Module):
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
    _replace_fmoe_layer(model, target, [0], False, router_name, experts_name, decoder_name, config)

    return get_fmoe_layers([], model, target)

def _replace_fmoe_layer(model, target, layer_idx, is_decoder, router_name, experts_name, decoder_name, config):
    for name, module in model.named_children():
        if name == decoder_name:
            is_decoder = True 
            layer_idx[0] = 0

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
                    layer_idx=layer_idx[0],
                    is_decoder=is_decoder,
                    num_expert=num_experts_per_worker,
                    d_model=config.d_model,
                    top_k=1,
                    world_size=world_size,
                    expert=lambda _: FMoESwitchTransformersDenseActDense(config),
                )

                # Move the weights over
                with torch.no_grad():
                    # Router
                    new_moe_layer.gate.gate.weight.copy_(router.classifier.weight)

                    # Experts
                    rank = dist.get_rank()
                    start_idx = rank * num_experts_per_worker
                    for i in range(num_experts_per_worker):
                        new_moe_layer.experts[i].wi.weight.copy_(experts[i+start_idx].wi.weight)
                        new_moe_layer.experts[i].wo.weight.copy_(experts[i+start_idx].wo.weight)

                layer_idx[0] += 1

                setattr(module, child_name, new_moe_layer)
            else:
                _replace_fmoe_layer(
                    child, 
                    target, 
                    layer_idx,
                    is_decoder,
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


################ DEEPSPEED #######################

import deepspeed

class DeepspeedMoE(deepspeed.moe.layer.MoE):
    def __init__(self, layer_idx=0, is_decoder=False, **kwargs):
        super().__init__(**kwargs)

        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        self.latencies = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def forward(self, hidden_states):
        self.start_event.record()
        output = super().forward(hidden_states)
        self.end_event.record()
        self.end_event.synchronize()
        self.latencies.append(self.start_event.elapsed_time(self.end_event))
        return output[0], (output[1], output[2])
    
    def save_statistics(self, DIR=""):
        path = f"{DIR}/moe"
        if self.is_decoder:
            path += "_decoder"
        path += f"_layer-{self.layer_idx}"

        with open(f"{path}.csv", "w") as f:
            fieldnames = ["iteration", "latency (ms)"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.latencies)):
                dic = {
                    "iteration": i,
                    "latency (ms)": self.latencies[i],
                }
             
                writer.writerow(dic)


def replace_deepspeed_layer(model, target, router_name, experts_name, decoder_name, config):
    _replace_deepspeed_layer(model, target, [0], False, router_name, experts_name, decoder_name, config)

    return get_deepspeed_layers([], model, target)

def _replace_deepspeed_layer(model, target, layer_idx, is_decoder, router_name, experts_name, decoder_name, config):
    for name, module in model.named_children():
        if name == decoder_name:
            is_decoder = True 
            layer_idx[0] = 0
        
        for child_name, child in module.named_children():
            if type(child).__name__ == target:
                router = getattr(child, router_name)
                experts = getattr(child, experts_name)
                if isinstance(experts, nn.ModuleDict):
                    experts = nn.ModuleList(experts.values())


                num_experts = len(experts)
                world_size = dist.get_world_size()
                num_experts_per_worker = num_experts // world_size

                new_moe_layer = DeepspeedMoE(
                    layer_idx=layer_idx[0],
                    is_decoder=is_decoder,
                    hidden_size=config.d_model,
                    expert=SwitchTransformersDenseActDense(config),
                    num_experts=num_experts,
                    ep_size=world_size,
                    k=1,
                )

                # Move the weights over
                with torch.no_grad():
                    # Router
                    new_moe_layer.deepspeed_moe.gate.wg.weight.copy_(router.classifier.weight)

                    # Experts
                    rank = dist.get_rank()
                    start_idx = rank * num_experts_per_worker
                    for i in range(num_experts_per_worker):
                        new_moe_layer.deepspeed_moe.experts.deepspeed_experts[i].wi.weight.copy_(experts[start_idx+i].wi.weight)
                        new_moe_layer.deepspeed_moe.experts.deepspeed_experts[i].wo.weight.copy_(experts[start_idx+i].wo.weight)

                layer_idx[0] += 1

                setattr(module, child_name, new_moe_layer)
            else:
                _replace_deepspeed_layer(
                    child, 
                    target, 
                    layer_idx,
                    is_decoder,
                    router_name, 
                    experts_name,
                    decoder_name,
                    config,
                )  

def get_deepspeed_layers(acc, model, target):
    for module in model.children():
        if isinstance(module, deepspeed.moe.layer.MoE):
            acc.append(module)
        else:
            acc = get_deepspeed_layers(acc, module, target)
    return acc