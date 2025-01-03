import pandas as pd 
import sys
import json
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(f"Usage: python3 {sys.argv[0]} x-axis-var path-to-runs")


data = []
for path in sys.argv[2:]:
    df = pd.read_csv(f"{path}/0/e2e.csv")
    with open(f"{path}/data.json") as f:
        meta_data = json.load(f)
        data.append({
            "y": df.sum(axis=0)["latency (s)"],
            "x": meta_data[sys.argv[1]]
        })

df = pd.DataFrame(data)
df = df.sort_values("x")
plt.plot(df["x"], df["y"])
plt.xlabel(sys.argv[1])
plt.ylabel("Run Duration (s)")

plt.savefig("tmp/duration_plot.png")