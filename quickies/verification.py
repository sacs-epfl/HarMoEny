import pandas as pd
import sys
import os
import json

if len(sys.argv) < 2:
    print(f"USAGE: python3 {sys.argv[0]} run_dir")
    exit(1)

path = sys.argv[1]
with open(os.path.join(path, "data.json")) as f:
    meta = json.load(f)

num_layers = 12
num_gpus = meta["world_size"]
batch_size = meta["batch_size"]
seq_len = meta["seq_len"]
num_layers_encoder = 6
num_toks_per_batch_encoder = batch_size * seq_len * num_gpus
num_toks_per_batch_decoder = batch_size * num_gpus

def verify_num_toks_processed():
    for layer_idx in range(num_layers):
        df = None
        for gpu_idx in range(num_gpus):
            _df = pd.read_csv(os.path.join(path, f"{gpu_idx}/moe_layer-{layer_idx}.csv"), index_col=0)
            _df = _df[["total number of tokens recv"]]
            if df is None:
                df = _df
            else:
                df["total number of tokens recv"] += _df["total number of tokens recv"]
        df = df.iloc[:-1]
        comparison = num_toks_per_batch_encoder if layer_idx < num_layers_encoder else num_toks_per_batch_decoder
        all_correct = (df["total number of tokens recv"] == comparison).all()
        if not all_correct:
            print(f"TOKENS LOST ISSUE: layer {layer_idx}")

def get_imbalance():
    df = None
    for layer_idx in range(num_layers_encoder):
        _df = None
        for gpu_idx in range(num_gpus):
            __df = pd.read_csv(os.path.join(path, f"{gpu_idx}/moe_layer-{layer_idx}.csv"), index_col=0)
            __df = __df[["total number of tokens recv"]]
            __df = __df.rename(columns={"total number of tokens recv": gpu_idx})
            if _df is None:
                _df = __df
            else:
                _df = _df.merge(__df, how="outer", on="iteration")
        _df = _df.iloc[:-1]
        _df["max"] = _df.max(axis=1)
        _df["min"] = _df.min(axis=1)
        _df["imbalance"] = (_df["max"] - _df["min"]) / _df["min"]
        _df = _df[["imbalance"]]
        _df = _df.rename(columns={"imbalance": layer_idx})
        
        if df is None:
            df = _df
        else:
            df = df.merge(_df, how="outer", on="iteration")
    print(df)

if __name__ == "__main__":
    verify_num_toks_processed()
    get_imbalance()