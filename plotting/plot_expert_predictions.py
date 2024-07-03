import pandas as pd 
import plotly.express as px
from theme import update_fig_to_theme

def load_file_pandas(file):
    return pd.read_csv(file)


def prediction_similarity(data):
    d = {}

    for index, row in data.iterrows():
        # Creating if missing
        if f"Layer {row['Layer Number']}" not in d:
            d[f"Layer {row['Layer Number']}"] = {}
            for column_name, value in row.items():
                if column_name in ["Batch Number", "Layer Number", "Token Number"]:
                    continue
                if int(column_name.split(" ")[1]) <= int(row["Layer Number"]): # We skip over any that are less than or equal then where it was captured the hiden units 
                    continue 
                if column_name not in d[f"Layer {row['Layer Number']}"]:
                    d[f"Layer {row['Layer Number']}"][column_name] = {"Hit": 0, "Miss": 0}
        
        # Populating
        for key, value in d[f"Layer {row['Layer Number']}"].items():
            if row[f"Layer {row['Layer Number']}"] == row[key]:
                value["Hit"] += 1
            else:
                value["Miss"] += 1
   
    df = pd.DataFrame(d)

    for index_row, row in df.iterrows():
        for column_name, value in row.items():
            if not isinstance(value, dict):
                continue
            df.at[index_row, column_name] = round((value["Hit"] / (value["Hit"] + value["Miss"])) * 100, 2)
    
    df = df.drop(columns=["Layer 11"])

    return df

def yandex(data):
    d = {}

    for index, row in data.iterrows():
        # Creating if missing
        if f"Layer {row['Layer Number']}" not in d:
            d[f"Layer {row['Layer Number']}"] = {}
            for column_name, value in row.items():
                if column_name in ["Batch Number", "Layer Number", "Token Number"]:
                    continue
                if int(column_name.split(" ")[1]) <= int(row["Layer Number"]): # We skip over any that are less than or equal then where it was captured the hiden units 
                    continue 
                if column_name not in d[f"Layer {row['Layer Number']}"]:
                    d[f"Layer {row['Layer Number']}"][column_name] = {"Hit": 0, "Miss": 0}

        # Populating
        for column_name, value in row.items():
            if column_name in ["Batch Number", "Layer Number", "Token Number"]:
                continue
            if int(column_name.split(" ")[1]) <= int(row["Layer Number"]): # We skip over any that are less than or equal then where it was captured the hiden units 
                    continue 
            e = data[(data["Batch Number"] == row["Batch Number"]) &
                (data["Token Number"] == row["Token Number"]) &
                (data["Layer Number"] == int(column_name.split(" ")[1]))
            ]

            if e.empty:
                continue

            if value == e[column_name].values[0]:
                d[f"Layer {row['Layer Number']}"][column_name]["Hit"] += 1
            else:
                d[f"Layer {row['Layer Number']}"][column_name]["Miss"] += 1
        
    df = pd.DataFrame(d)

    for index_row, row in df.iterrows():
        for column_name, value in row.items():
            if not isinstance(value, dict):
                continue
            df.at[index_row, column_name] = round((value["Hit"] / (value["Hit"] + value["Miss"])) * 100, 2)
    
    df = df.drop(columns=["Layer 11"])

    return df

def clean_data(data):
    data = data.drop(data[data["Batch Number"] == 0].index)
    return data


def plot_heatmap(df, reverse=True, file_name="tmp", x_axis="", y_axis="", legend=""):
    if reverse:
        df = df.iloc[::-1]
    fig = px.imshow(df, text_auto=True, labels=dict(x=x_axis, y=y_axis, color=legend), zmin=0, zmax=100)
    update_fig_to_theme(fig)
    fig.write_image(f"plots/{file_name}.png")


def main():
    df = load_file_pandas("outputs/hidden_states_gate_choices-bookcorpus-medium.csv")
    df = clean_data(df)
    df_similarity = prediction_similarity(df)
    plot_heatmap(df_similarity, file_name="medium-similarity", x_axis="Ground Truth Layer", y_axis="Compared To Layer", legend="Prediction Similarity (%)")
    df_yandex = yandex(df)
    plot_heatmap(df_yandex, file_name="medium-yandex", x_axis="Layer Hidden Units Taken From", y_axis="Layer Hidden Units Given To Gate", legend="Accuracy (%)")

if __name__ == "__main__":
    main()