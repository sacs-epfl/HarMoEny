import pandas as pd
import plotly.express as px 
from theme import update_fig_to_theme

files = [{"path": "outputs/naive.csv", "name": "DeepSpeed (naive)"}, 
{"path": "outputs/ideal-greedy.csv", "name": "Ideal (greedy)"},
{"path": "outputs/demeter.csv", "name": "Demeter"},
{"path": "outputs/demeter-dp-3.csv", "name": "Demeter (dp-3)"}]

CACHE = {}

def load_file_pandas(file_name):
    return pd.read_csv(file_name)

def plot_files(files, col="Layer 11", file_name="test.png", num_batches=None):
    dfs = []
    for file in files:
        if file["name"] in CACHE:
            dfs.append(CACHE[file["name"]])
        else:
            df = load_file_pandas(file["path"])
            avg = df.drop(columns=["Batch Number"]).mean(axis=1)
            df["Average Layer"] = avg
            df["Method"] = file["name"]
            dfs.append(df)
            CACHE[file["name"]] = df
    
    df = pd.concat(dfs, ignore_index=True)
    
    if num_batches:
        df = df[df["Batch Number"] <= num_batches]

    fig = px.line(df, x="Batch Number", y=col, color="Method")
    update_fig_to_theme(fig)
    fig.update_xaxes(dtick=10)
    fig.update_yaxes(title=dict(text=f"{col} Imbalance Score"))

    # Hide items
    # fig.update_layout(showlegend=False, 
    #     title=dict(text=col, font=dict(family="Fira Sans", color="#000000", size=42)),
    #     xaxis_title='', 
    #     yaxis_title='', 
    #     xaxis=dict(
    #         # showticklabels=False
    #     ),
    #     yaxis=dict(
    #         # showticklabels=False,
    #         range=[0, 35]
    #     )
    # )

    fig.write_image(f"plots/{file_name}-{'_'.join(col.split(' '))}-b{num_batches}.png")


def main():
    # plot_files(files[:2], file_name="no-demeter", col="Average Layer", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 1", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 3", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 5", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 7", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 9", num_batches=100)
    # plot_files(files[:2], file_name="no-demeter", col="Layer 11", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Average Layer", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 1", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 3", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 5", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 7", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 9", num_batches=100)
    # plot_files(files[:3], file_name="demeter", col="Layer 11", num_batches=100)
    plot_files(files, file_name="demeter-w-dp", col="Average Layer", num_batches=100)


if __name__ == "__main__":
    main()