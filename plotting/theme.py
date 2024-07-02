import plotly.express as px

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def update_fig_to_theme(fig, title=None, xaxis=None, yaxis=None):
    initial_layout_updates = {}

    if title is not None:
        initial_layout_updates["title"] = title
    
    if xaxis is not None:
        initial_layout_updates["xaxis_title"] = xaxis

    if yaxis is not None:
        initial_layout_updates["yaxis_title"] = yaxis

    additional_layout_updates = {
        "font": dict(family="Fira Sans", color="#000000"),
        "titlefont": dict(family="Fira Sans", color="#000000", size=42),
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "xaxis": dict(showgrid=True, gridcolor="#e5e5e5", color="#000000", titlefont=dict(size=28), tickfont=dict(size=24)),
        "yaxis": dict(showgrid=True, gridcolor="#e5e5e5", color="#000000", titlefont=dict(size=28), tickfont=dict(size=24)),
        "legend": dict(font=dict(color="#000000", size=28)),   
        "width": 1280,
        "height": 720,
    }

    fig.update_layout(**initial_layout_updates, **additional_layout_updates)

    for i in range(len(fig.data)):
        if "line" in fig.data[i]:
            fig.data[i].line.color = colours[i]