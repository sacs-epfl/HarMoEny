import sys
import pandas as pd

if len(sys.argv) < 3:
    print("You need to provide at least two paths to compare")


for path in sys.argv[1:]:
    df = pd.read_csv(f"{path}/0/e2e.csv")
    df = df.sum(axis=0)
    print(f"{path}: {df['latency (s)']}")
