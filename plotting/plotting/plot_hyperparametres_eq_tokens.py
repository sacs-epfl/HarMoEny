import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import numpy as np
from utils import save_plot

def main():
    df = pd.read_csv("../data_processed/hyperparametres/eq_tokens/eq_tokens_duration.csv", index_col=0)
    df = df.sort_values(by=["eq_tokens"])
    
    plt.plot(df["eq_tokens"], df["duration (s)"])

    save_plot(plt, "../figures/hyperparameters/eq_tokens/eq_tokens_duration.png")

if __name__ == "__main__":
    main()