import pandas as pd

FILE = "/Users/juridirocco/development/ghaction/results/code_statements/Fold_5_Results.csv"
df = pd.read_csv(FILE)


for i, row in df.iterrows():
    if (pd.isna(row['Learning_label'])):
        print(row)