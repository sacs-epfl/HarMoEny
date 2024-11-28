import sys
import pandas as pd
import sys

if len(sys.argv) < 3:
    print("Please provide at least two comparisons")
    exit(0)

num_layers = 12

paths = sys.argv[1:]
df = pd.DataFrame()
for i in range(num_layers):
    _df = None
    for path in paths:
        __df = pd.read_csv(f"{path}/0/moe_layer-{i}.csv", index_col=0)
        __df = __df[["latency (ms)"]]
        __df = __df.rename(columns={"latency (ms)": path})
        if _df is None:
            _df = __df
        else:
            _df = _df.merge(__df, on="iteration", how="outer")
    df = pd.concat([df, _df])

print(df)
print(df.sum(axis=0))