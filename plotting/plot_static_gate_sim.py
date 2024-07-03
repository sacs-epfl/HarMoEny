import pandas as pd 
import plotly.express as px
from theme import update_fig_to_theme

def load_file_pandas(file):
    return pd.read_csv(file, index_col="Layer")


def plot_heatmap(df, reverse=True, file_name="tmp", x_axis="", y_axis="", legend=""):
    if reverse:
        df = df.iloc[::-1]
    fig = px.imshow(df, text_auto=True, labels=dict(x=x_axis, y=y_axis, color=legend), zmin=-1, zmax=1)
    update_fig_to_theme(fig)
    fig.write_image(f"plots/{file_name}.png")


def main():
    df = load_file_pandas("outputs/switch-transformer-8-gate-cosine-similarity.csv")
    df = df.drop(columns="Layer 1")
    df = df.drop(index="Layer 11")
    df = round(df, 4)
    print(df)
    plot_heatmap(df, file_name="switchtransformer-8-cosine-gate-sim", legend="Cosine Similarity")

if __name__ == "__main__":
    main()