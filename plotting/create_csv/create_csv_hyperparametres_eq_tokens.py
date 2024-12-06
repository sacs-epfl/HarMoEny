from utils import save_pd, Data
import matplotlib.pyplot as plt 
import pandas as pd
import ast
import os

directory_path = "../data/hyperparametres/eq_tokens"
data = Data(directory_path)

df = []
for child in data.get_children():
    _df = data.load(f"{child}/0/e2e.csv")
    _df = _df.sum(axis=0)
    df.append({
        "eq_tokens": int(child),
        "duration (s)": _df["latency (s)"]
    })
df = pd.DataFrame(df)
save_pd(df, "../data_processed/hyperparametres/eq_tokens/eq_tokens_duration.csv")