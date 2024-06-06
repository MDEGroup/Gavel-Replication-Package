import pandas as pd


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def compute_score(path):
    f1 = pd.read_csv(path)
    column_prediction = [x for x in f1.columns if x.endswith('_label')]
    column_label = [x for x in f1.columns if x.endswith('_prediction')]
    for i,c in enumerate(column_prediction):
        # f1[c] = f1[c].apply(lambda x: x.replace("[","").replace("]","").replace(" ",""))
        f1[c] = f1[c].apply(lambda x: int(x.replace("tensor(","").replace(")","").replace(" ","")))
        # count the number of rows with value different from 0
        print(f1[c].value_counts())
        print(f1[column_label[i]].value_counts())
        print("========================================")

BASE_PATH = '/Users/juridirocco/development/ghaction/results/all/Fold_'
for i in range(1, 6):
    path = f'{BASE_PATH}{i}_Results.csv'
    compute_score(path)
    break