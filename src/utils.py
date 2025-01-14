import csv

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersDenseActDense


class TimedModule(nn.Module):
    def __init__(self, child, idx=0):
        super().__init__()

        self.child = child
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.latencies = []
        self.idx = idx 
    
    def forward(self, x):
        self.start.record()
        x = self.child(x)
        self.end.record()
        self.end.synchronize()
        self.latencies.append(self.start.elapsed_time(self.end))
        return x
    
    def get_latencies(self):
        return self.latencies[:]

def get_timing_modules(acc, module):
    if type(module).__name__ == "TimedModule":
        acc.append(module)
    else:
        for child in module.children():
            acc = get_timing_modules(acc, child)
    return acc 