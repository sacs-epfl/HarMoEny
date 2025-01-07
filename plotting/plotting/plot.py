import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import colormaps
import re
import os

def make_path_safe(x):
    x = x.lower()
    x = re.sub(r'[<>:"/\\|?*]', '_', x)
    x = re.sub(r'\s+', '_', x)
    return x

def parse_args():
    parser = argparse.ArgumentParser(description="Plot a dataframe based on x-axis")
    parser.add_argument(
        '--x-axis',
        required=True,
        help="Dataframe column to plot as x-axis"
    )
    parser.add_argument(
        "--y-axis",
        required=True,
        help="Dataframe column to plot as y-axis"
    )
    parser.add_argument(
        "--grouping",
        help="Dataframe column to group by"
    )
    parser.add_argument(
        '--path',
        required=True,
        help="Path to dataframe"
    )
    parser.add_argument(
        "--title",
        help="Title of the plot, will also be used as file name"
    )
    return parser.parse_args()

def plot(args):
    df = pd.read_csv(args.path, index_col=0)
    if args.grouping:
        df = df[[args.x_axis, args.y_axis, args.grouping]]
    else:
        df = df[[args.x_axis, args.y_axis]]
    
    if args.grouping:
        df = df.sort_values(args.grouping)
        df = df.pivot(index=args.x_axis, columns=args.grouping, values=args.y_axis)
        
        bar_width = 0.2
        grouping_gap = 0.5 
        num_groupings = len(df.columns)
        x_positions = []
        current_x = 0
        
        for _ in range(len(df.index)):
            arr = np.arange(current_x, current_x + num_groupings * bar_width, bar_width)
            x_positions.append(arr)
            current_x = arr[-1] + grouping_gap

        x_positions = np.array(x_positions)

        fig, ax = plt.subplots(figsize=(12, 5))

        color_palette = colormaps.get_cmap("tab10")  
        grouping_colors = {group: color_palette(i) for i, group in enumerate(df.columns)}

        for i, (x_axis, positions) in enumerate(zip(df.index, x_positions)):
            for j, group in enumerate(df.columns):
                ax.bar(positions[j], df.loc[x_axis, group], width=bar_width, color=grouping_colors[group], label=group if i == 0 else "")

        # Flatten x_positions for xticks and add gaps for visualization
        flat_x_positions = [positions.mean() for positions in x_positions]

        ax.set_xticks(flat_x_positions)
        ax.set_xticklabels(df.index)
        ax.set_xlabel(args.x_axis)
        ax.set_ylabel(args.y_axis)
        ax.legend(title=args.grouping, loc="lower right")

        if args.title:
            ax.set_title(args.title)
    else:
        print("HERE")

def main():
    args = parse_args()
    plot(args)
    filename = "fig"
    if args.title:
        filename = make_path_safe(args.title)
    filename += ".png"

    plt.savefig(os.path.join("../figures", filename))

if __name__ == "__main__":
    main()