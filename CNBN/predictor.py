import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, FeatureHasher
from plot_auc_curve import plot_auc_curve

import operator
import csv

from sklearn.metrics import precision_score, recall_score, f1_score


def optimize_hps(train_data, label_data):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True,
                                  analyzer='word', encoding='utf-8',
                                  token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                  max_features=5000)),
        ('clf', LinearSVC())  # Using LinearSVC as an example
    ])

    # Hyperparameters to finetune
    parameters = {
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__C': (0.1, 1, 10)
    }

    # Grid search across our parameters
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, cv= cv)
    grid_search.fit(train_data, label_data)

    best_model = grid_search.best_estimator_
    print(best_model)
    return best_model




def train(data):
    ###Extracting features
    #print('Extracting features from dataset')
    count_vect = TfidfVectorizer(input='train', stop_words={'english'}, lowercase=True, token_pattern=r'\w{1,}',
                                 analyzer='word', encoding='utf-8', ngram_range=(1, 2), max_features=5000)

    train_data = []
    label_data = []

    for label in data:
        texts = data.get(label)

        for text in texts:
            train_data.append(text)
            label_data.append(label)

    train_vectors = count_vect.fit_transform(label_data)
    train_vectors.shape
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_vectors)

    train_tfidf.shape

    #optimize_hps(train_data,label_data)

    model = ComplementNB()

    model.fit(train_tfidf, train_data)

    return model, tfidf_transformer, count_vect

def predict(test,model,tfidf_transformer,count_vect):
    out_dict = {}
    X_new_counts = count_vect.transform([test])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    #model.predict(X_new_tfidf)
    ranked_dict = {}

    for prob in model.predict_proba(X_new_tfidf):
        for cat, p in zip(model.classes_, prob):
            # print(cat+":"+str(p))
            out_dict.update({cat: str(p)})
            ranked_dict = sorted(out_dict.items(), key=operator.itemgetter(1), reverse=True)

    #print (ranked_dict)
    predicted_topics = []
    for tuple in ranked_dict:
        predicted_topics.append(tuple[0])

    return predicted_topics


def compute_precision(predicted, actual):
    if not actual:
        return 0.0

    true_positives = len([value for value in predicted if value in actual])
    false_positives = len([value for value in predicted if value not in actual])

    if true_positives + false_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    return precision * 100


def compute_recall(predicted, actual):
    if not actual:
        return 0.0

    true_positives = len([value for value in predicted if value in actual])
    false_negatives = len([value for value in actual if value not in predicted])

    if true_positives + false_negatives == 0:
        return 0.0

    recall = true_positives / (true_positives + false_negatives)
    return recall * 100


def compute_f1(pr, rec):
    if pr + rec == 0.0:
        return 0.0
    else:
        return 2 * (pr * rec) / (pr + rec)

def compute_success_rate(predicted,actual,n):
    if actual:
        match = [value for value in predicted if value in actual]
        if len(match) >= n:
            return 1
        else:
            return 0
    else:
        return 0


def compute_metrics(predicted, actual):


    succ_rate = compute_success_rate(predicted,actual,1)
    pr = compute_precision(predicted,actual)
    rec = compute_recall(predicted, actual)
    f1 = compute_f1(pr, rec)

    return succ_rate, pr, rec, f1


# def filter_topics_level(actual, predicted, dict_topics, level):
#     filter_actual = []
#     filter_predicted = []
#
#     for actual_topic in actual:
#         print("dict level ",dict_topics.get(actual_topic))
#         print("actual level ", level)
#         if dict_topics.get(actual_topic) == level:
#             filter_actual.append(actual_topic)
#
#     for predicted_topic in predicted:
#         if dict_topics.get(predicted_topic) == level:
#             filter_predicted.append(predicted_topic)
#
#     return filter_actual, filter_predicted


def compute_metrics_multi_cat(predicted, actual, categories, out_file):
    metrics = {}
    for category in categories:
        succ_rate = compute_success_rate_cat(predicted, actual, category,1)
        pr = compute_precision_cat(predicted, actual, category)
        rec = compute_recall_cat(predicted, actual, category)
        f1 = compute_f1(pr, rec)

        metrics[category] = (succ_rate, pr, rec, f1)

        with open(out_file, 'w', newline='', encoding='utf-8', errors='ignore') as csvfile:
            fieldnames = ['Category', 'Success Rate', 'Precision', 'Recall', 'F1 Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for cat, values in metrics.items():
                writer.writerow(
                    {'Category': cat, 'Success Rate': values[0], 'Precision': values[1],
                     'Recall': values[2], 'F1 Score': values[3]})

    return metrics


def compute_precision_cat(predicted, actual, category):
    if not actual:
        return 0.0

    true_positives = len([value for value in predicted if value == category and value in actual])
    false_positives = len([value for value in predicted if value == category and value not in actual])

    if true_positives + false_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    return precision * 100


def compute_recall_cat(predicted, actual, category):
    if not actual:
        return 0.0

    true_positives = len([value for value in predicted if value == category and value in actual])
    false_negatives = len([value for value in actual if value == category and value not in predicted])

    if true_positives + false_negatives == 0:
        return 0.0

    recall = true_positives / (true_positives + false_negatives)
    return recall * 100


def compute_success_rate_cat(predicted, actual, category, n):
    if actual:
        match = [value for value in predicted if value == category and value in actual]
        if len(match) >= n:
            return 1
        else:
            return 0
    else:
        return 0
