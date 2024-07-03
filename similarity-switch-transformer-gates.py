import os 
os.environ["HF_HOME"] = "/cache"

from transformers import AutoModel
import json 
import csv
import torch
import torch.nn.functional as F


model = AutoModel.from_pretrained("google/switch-base-8")

gates = [None] * len(model.encoder.block)

for idx_b, block in enumerate(model.encoder.block):
    if hasattr(block.layer[1].mlp, "router"):
        gates[idx_b] = block.layer[1].mlp.router.classifier.weight


gate_sim = {}

for i in range(len(gates)):
    if gates[i] is None:
        continue
    if i not in gate_sim:
        gate_sim[i] = {}
    for j in range(i+1, len(gates)):
        if gates[j] is None:
            continue
        gate_sim[i][j] = F.cosine_similarity(gates[i].view(-1).unsqueeze(0), gates[j].view(-1).unsqueeze(0)).item()


# Reorganize
gate_sim_organized = []

for key, value in gate_sim.items():
    d = {}
    d["Layer"] = f"Layer {key}"
    for k2, v2 in value.items():
        d[f"Layer {k2}"] = v2
    gate_sim_organized.append(d)


with open("outputs/switch-transformer-8-gate-cosine-similarity.csv", "w") as w:
    fieldNames = ["Layer", "Layer 1", "Layer 3", "Layer 5", "Layer 7", "Layer 9", "Layer 11"]
    writer = csv.DictWriter(w, fieldNames) 
    writer.writeheader()
    writer.writerows(gate_sim_organized)


