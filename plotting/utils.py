import os
import pandas as pd
import glob
import json

def save_pd(df, path: str):
    parent_dir = os.path.dirname(path)
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    df.to_csv(path)

def save_plot(plt, path: str):
    parent_dir = os.path.dirname(path)
    
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    plt.savefig(path)
    plt.close()

class Data():
    def __init__(self, directory_path):
        self.most_recent_folder = Data.get_most_recent_folder(directory_path)

    def get_most_recent_folder(directory_path):
        subdirs = [d for d in glob.glob(os.path.join(directory_path, '*')) if os.path.isdir(d)]
        
        if subdirs:
            most_recent_folder = max(subdirs, key=os.path.getmtime)
            return most_recent_folder
        else:
            return None

    def load(self, name):
        df = pd.read_csv(f"{self.most_recent_folder}/{name}", index_col=0)
        return df
    
    def read_meta(self, path):
        with open(f"{self.most_recent_folder}/{path}/data.json", "r") as f:
            return json.load(f)
