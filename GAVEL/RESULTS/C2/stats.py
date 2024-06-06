import pandas as pd
import classification_report2latex

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def compute_score(f1):
    
    column_prediction = [x for x in f1.columns if x.endswith('_label')]
    column_label = [x for x in f1.columns if x.endswith('_prediction')]
    prediction_list = []
    ground_truth_list = []
    for i, row in f1.iterrows():
        prediction = []
        ground_truth = []
        for c in column_prediction:
            if row[c] == 'tensor(1)':
                prediction.append(c.replace("_label",""))
        for c in column_label:
            if row[c] == 1:
                ground_truth.append(c.replace("_prediction",""))
        prediction_list.append(prediction)
        ground_truth_list.append(ground_truth)
    prediction = []
    ground_truth = []
    for c in column_prediction:
        prediction.append(c.replace("_label",""))
    for c in column_label:
        ground_truth.append(c.replace("_prediction",""))
    prediction_list.append(prediction)
    ground_truth_list.append(ground_truth)
    mlb = MultiLabelBinarizer()
    pred = mlb.fit_transform(prediction_list)
    klb = MultiLabelBinarizer()
    lbl = klb.fit_transform(ground_truth_list)
    with open("result.tex", 'w') as f:
        data = classification_report2latex.parse_classification_report(classification_report(lbl, pred, target_names=klb.classes_))
        f.write(classification_report2latex.report_to_latex_table(data))
    



BASE_PATH = '/Users/juridirocco/development/ghaction/results/code/Fold_'
result = pd.DataFrame()
for i in range(1, 6):
    path = f'{BASE_PATH}{i}_Results.csv'
    result = pd.concat([result, pd.read_csv(path)])

compute_score(result)


    
    


