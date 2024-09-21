import os 
os.environ["HF_HOME"] = "/cache"

from transformers import AutoModel
import json 
import csv
import torch

model = AutoModel.from_pretrained("google/switch-base-8")

gates = [None] * len(model.encoder.block)

for idx_b, block in enumerate(model.encoder.block):
    if hasattr(block.layer[1].mlp, "router"):
        gates[idx_b] = block.layer[1].mlp.router.to("cuda")


with open("data/hidden_states-bookcorpus-medium.json", "r") as f:
    data = json.load(f)
    token_choices_list = []
    for batch in data:
        for idx_t, token in enumerate(batch["Hidden_States"][0]):
            token = torch.tensor([token]).to("cuda")
            entry = {}
            entry["Batch Number"] = batch["Batch"]
            entry["Token Number"] = idx_t
            entry["Layer Number"] = batch["Layer"]
            for idx_ga, gate in enumerate(gates):
                if gate is not None:
                    router_mask, router_probs, router_logits = gate(token)
                    entry[f"Layer {idx_ga}"] = torch.argmax(router_mask, dim=-1)[0].to("cpu").item()
            token_choices_list.append(entry)

    with open("outputs/hidden_states_gate_choices-bookcorpus-medium.csv", "w") as w:
        fieldNames = ["Batch Number", "Token Number", "Layer Number", "Layer 1", "Layer 3", "Layer 5", "Layer 7", "Layer 9", "Layer 11"]
        writer = csv.DictWriter(w, fieldNames)
        writer.writeheader()
        writer.writerows(token_choices_list)