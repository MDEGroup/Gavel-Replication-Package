import pandas as pd
import classification_report2latex

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def compute_score(f1):
    
    
    
    prediction_list = []
    ground_truth_list = []
    pred_set = set()
    gt_set = set()
    f1['actual'].apply(lambda x: x.split(','))
    f1['predicted'].apply(lambda x: x.split(',')[:2])
    for i, row in f1.iterrows():
        prediction = []
        ground_truth = []
        prediction = row['predicted'].split(",")
        ground_truth = row['actual'].split(",")
        for p in prediction:
            pred_set.add(p)
        for g in ground_truth:
            gt_set.add(g)
        prediction_list.append(prediction)
        ground_truth_list.append(ground_truth)
    union_set = list(pred_set.union(gt_set))
    print(union_set)
    prediction_list.append(union_set)
    ground_truth_list.append(union_set)
    mlb = MultiLabelBinarizer()
    pred = mlb.fit_transform(prediction_list)
    lbl = mlb.fit_transform(ground_truth_list)
    with open("result.tex", 'w') as f:
        data = classification_report2latex.parse_classification_report(classification_report(lbl, pred, target_names=mlb.classes_))
        f.write(classification_report2latex.report_to_latex_table(data))
    



BASE_PATH = '/Users/juridirocco/development/ghaction/results/cnb/results_complete_all_round_'
result = pd.DataFrame()
for i in range(1, 6):
    path = f'{BASE_PATH}{i}.csv'
    result = pd.concat([result, pd.read_csv(path)])
result.reset_index(drop=True, inplace=True)

compute_score(result)



    
    


