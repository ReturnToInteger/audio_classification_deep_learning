import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pig_detection import get_files, plot_image_square
import argparse

parser = argparse.ArgumentParser(description="Plot pig noise detection results.")
parser.add_argument("--folder", type=str, default=None, help="Path to the folder containing detection results.")
args = parser.parse_args()
if not args.folder:
    raise ValueError("Please provide the path to the folder containing detection results using --folder argument.")
path = args.folder

data_list=[]
for txt in get_files(path):
    date=txt.split('_')[1]
    data= pd.read_csv(os.path.join(path,txt),sep="\t", header=None)
    data["date"]=date
    data_list.append(data)

df=pd.concat(data_list)
df.columns = ["start", "end", "label", "date"]
df["duration"]=df["end"]-df["start"]

df_grouped = df.groupby(["date", "label"])["duration"].sum().unstack(fill_value=0).reset_index(inplace=False)

df_grouped.plot(x="date", y=[0,1,2,3], kind="line", figsize=(10,5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()