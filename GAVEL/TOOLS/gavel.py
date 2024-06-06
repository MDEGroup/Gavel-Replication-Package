import pandas as pd
import json
import os
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch

BASE_MODEL = "BAAI/bge-base-en-v1.5"
RANDOM_SEED = 42
OUTPUT_PATH = 'output'

#READ DATA
df = pd.read_csv("code_block.csv")
df = df.dropna()
df = df.reset_index(drop=True)
df["text"] = df["readme"]
df.drop(columns=["id", "readme"], axis=1, inplace=True)
target_list = list([col for col in df.columns if col!="text"])

#FROM MULTI COLUMN CLASSES TO LIST OF CLASSES
def prepare_multilabel_label_col(row, targets):
    values = [row[target] for target in targets]
    return values

import torch
training_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(training_device)
df_copy = df.copy()
df_copy["label"] = df_copy.apply(lambda x: prepare_multilabel_label_col(x, target_list), axis=1)
df_copy.drop(columns=target_list, axis=1, inplace=True)
print("DATA PROCESSED")
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    multi_target_strategy="one-vs-rest",
)
model.to("cuda")

print("PRETRAINED MODEL PROCESSED")
def get_prediction_for_each_label(predictions, target_list):
    values = {}
    for i, target in enumerate(target_list):
        values["{0}_prediction".format(target)] = predictions[i]
    return values

def get_true_value_for_each_label(label, target_list):
    values = {}
    for i, target in enumerate(target_list):
        values["{0}_label".format(target)] = label[i]
    return values

def get_test_data_with_complete_values(test_df, y_pred, target_list):
    updated_test_data = []
    count = 0
    for i, row in test_df.iterrows():
        predictions = get_prediction_for_each_label(row["label"], target_list)
        row_labels = get_true_value_for_each_label(y_pred[count], target_list)
        for target in target_list:
            row["{0}_prediction".format(target)] = predictions["{0}_prediction".format(target)]
            row["{0}_label".format(target)] = row_labels["{0}_label".format(target)]
        updated_test_data.append(row)
        count += 1
    test_data = pd.DataFrame(updated_test_data)
    return test_data

import json
from sklearn.metrics import accuracy_score

def get_all_and_average_accuracy(test_data, target_list):
    accuracy_data = {}
    accuracy_scores = []
    for target in target_list:
        preds = test_data["{0}_prediction".format(target)].to_list()
        labels = test_data["{0}_label".format(target)].to_list()
        accuracy = accuracy_score(labels, preds)
        accuracy_data[target] = accuracy
        accuracy_scores.append(accuracy)
    accuracy_data = json.dumps(accuracy_data, indent = 4)
    average_accuracy_score = sum(accuracy_scores)/len(accuracy_scores)*100
    return accuracy_data, average_accuracy_score

def train_and_evaluate_model(train_set, test_set):
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_set,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=4,
        num_epochs=5,
        num_iterations=20,
    )
    start_time = datetime.now()
    trainer.train()
    end_train = datetime.now()
    print(f"Training time: {(end_train - start_time).total_seconds()}")
    y_pred = trainer.model.predict(test_set["text"])
    print(f"Testing time: {(datetime.now() - end_train).total_seconds()}")
    return y_pred

import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
kf = KFold(n_splits=5)

X = df_copy["text"]
y = df_copy["label"]

fold = 1
fold_accuracy_values = []
for train_index, test_index in kf.split(X, y):
    start_time = datetime.now()
    print(f"STARTING FOLD {fold}")
    train_df = df_copy.loc[train_index]
    test_df = df_copy.loc[test_index]
    train_set = Dataset.from_pandas(train_df)
    test_set = Dataset.from_pandas(test_df)
    y_pred = train_and_evaluate_model(train_set, test_set)
    updated_test_df = get_test_data_with_complete_values(test_df, y_pred, target_list)
    updated_test_df.to_csv(f"Fold_{fold}_Results.csv")
    accuracy_data, average_accuracy = get_all_and_average_accuracy(updated_test_df, target_list)
    print("Accuracy data for Fold ", fold)
    print(json.dumps(accuracy_data, indent = 4))
    print("Average accuracy score for all labels: ", average_accuracy)
    fold_accuracy_values.append(average_accuracy)
    time_difference = (datetime.now() - start_time).total_seconds()
    print(f"Execution Time for {fold}: {time_difference} s")
    fold = fold + 1
    
average_cross_fold_accuracy = (sum(fold_accuracy_values)/ len(fold_accuracy_values))
print("Average cross-fold accuracy: ", average_cross_fold_accuracy)

print("Done!")