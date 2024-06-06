import pandas as pd
from collections import defaultdict
import numpy as np

from sklearn.model_selection import KFold,StratifiedKFold, train_test_split


import marko


def remove_nan_values(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    df_new = df.dropna()
    df_new.to_csv(out_csv, index=False)

def merge_csv(file1, file2, output_file):
    """
    Merge two CSV files with the same structure.

    Parameters:
    - file1 (str): Path to the first CSV file.
    - file2 (str): Path to the second CSV file.
    - output_file (str): Path to save the merged CSV file.
    """
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Concatenate the DataFrames along the rows
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)






def create_5_fold_splits(df, target_column):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        train_X = train_df.drop(columns=[target_column])
        train_y = train_df[target_column]
        test_X = test_df.drop(columns=[target_column])
        test_y = test_df[target_column]

        train_X.to_csv(f'train_X_fold_{fold}.csv', index=False)
        train_y.to_csv(f'train_y_fold_{fold}.csv', index=False)
        test_X.to_csv(f'test_X_fold_{fold}.csv', index=False)
        test_y.to_csv(f'test_y_fold_{fold}.csv', index=False)

        fold += 1



def read_data(csv_file):
    df_readme=pd.read_csv(csv_file, sep=',')
    df_readme.dropna(inplace=True)
    dict_readme = {}
    dict_project = {}

    for index, row in df_readme.iterrows():
        if row['readme'] :
            cat_list = []
            for key, value in row[2:].to_dict().items():

                if value == 1 and key not in cat_list:
                    cat_list.append(key)

            dict_readme.update({row['readme']: cat_list})
            dict_project.update({row['readme']: row['id']})


    df_new = pd.Series(dict_readme)
    training, testing = [i.to_dict() for i in train_test_split(df_new, train_size=0.8)]

    return training, testing, dict_project





def read_data_single(csv_file):
    df_readme=pd.read_csv(csv_file, sep=',')
    df_readme.dropna(inplace=True)
    dict_readme = {}
    dict_project = {}

    for index, row in df_readme.iterrows():
        if row['readme'] :
            cat_list = []
            for key, value in row[2:].to_dict().items():

                if value == 1 and key not in cat_list:
                    cat_list.append(key)

            dict_readme.update({row['readme']: cat_list})
            dict_project.update({row['readme']: row['id']})


    df_new = pd.Series(dict_readme)


    return df_new.to_dict(), dict_project



def split_data_round(path):


    df = pd.read_json(path, lines=True)
    msk = np.random.rand(len(df)) < 0.9
    df_train = df[msk]
    df_test = df[~msk]

    print(df_train)
    print(df_test)
    train_dict = defaultdict(list)
    dict_levels = create_levels_dict(df_train['labels'], df_train['levels'])
    test_dict = {}

    for desc, labels in zip(df_train['readme_text'], df_train['labels']):

        for l in labels:
            train_dict[l].append(desc)


    for desc, labels in zip(df_test['readme_text'], df_test['labels']):
       test_dict.update({desc: labels})

    return train_dict, test_dict, dict_levels



def create_levels_dict(topics, levels):


    levels_dict = {}
    #levels_dict = dict(zip_iterator)

    for t, l in zip(topics,levels):
        for t1, l1 in zip(t,l):
            levels_dict.update({t1 : l1})



    print(levels_dict)
    print()


    return levels_dict

# def computes_avg_metrics(results_file):
#     column_names = ['success_rate','precision','recall','f1']
#     df_results = pd.read_csv(results_file, names=column_names)
#     #df_half = df_results.iloc[75:,:]
#     avg_success = df_results['succ'].mean()
#     avg_pr = df_results['pr'].mean()
#     avg_rec = df_results['rec'].mean()
#     avg_f1 = df_results['f1'].mean()
#     #avg_time = df_half['time'].mean()

#    return avg_pr, avg_rec, avg_f1, avg_success

def print_metrics(path, filename):
    sum_succ = 0
    sum_pr = 0
    sum_rec = 0
    sum_f1 = 0

    for i in range(1, 11):
        succ, pr, rec, f1 = compute_avg_metrics(path + filename + str(i) + '.csv')

        print(succ, pr, rec, f1)
        sum_succ += succ
        sum_pr += pr
        sum_rec += rec
        sum_f1 += f1
        #sum_time += time

        # cosine_sum_succ += succ_cosine
        # cosine_sum_pr += pr_cosine
        # cosine_sum_rec += rec_cosine
        # cosine_sum_f1 += f1_cosine
        #
        # lev_sum_succ += succ_lev
        # lev_sum_pr += pr_lev
        # lev_sum_rec += rec_lev
        # lev_sum_f1 += f1_lev

    print('std metrics')
    print(sum_succ / 10)
    print(sum_pr / 10)
    print(sum_rec / 10)
    print(sum_f1 / 10)

def compute_avg_metrics(csv_results):
    column_names = ['success_rate','precision','recall','f1']
    df_results = pd.read_csv(csv_results,sep=',')
    avg_success = df_results['success_rate'].mean()
    avg_pr = df_results['precision'].mean()
    avg_rec = df_results['recall'].mean()
    avg_f1 = df_results['f1'].mean()

    return avg_success,avg_pr, avg_rec, avg_f1

def split_csv_file(file, n_splits=5, train_size=0.8):
    # Load the CSV file
    df = pd.read_csv(file)

    # Initialize the KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=False)

    # Split the indices
    fold = 1
    for train_index, test_index in kf.split(df):
        # Calculate the actual size for train and test splits
        train_index = train_index[:int(len(train_index) * train_size)]
        test_index = test_index[:int(len(test_index) * (1 - train_size))]

        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        # Save the folds
        df_train.to_csv(f'{file.split(".")[0]}_train_fold_{fold}.csv', index=False)
        df_test.to_csv(f'{file.split(".")[0]}_test_fold_{fold}.csv', index=False)

        fold += 1
def draw_times():


    # Data
    configurations = ['C1', 'C5', 'C2', 'C3', 'C4']
    cnb_testing = [0.51, 0.43, 0.34, 0.07, 0.26]
    cnb_training = [1.82, 1.34, 0.39, 0.04, 0.30]
    gavel_testing = [11.47, 10.34, 7.40, 1.84, 6.95]
    gavel_training = [58939.06, 58790.51, 49598.60, 9951.64, 48020.35]

    x = np.arange(len(configurations))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, cnb_testing, width, label='CNB')
    rects2 = ax.bar(x + width/2, gavel_testing, width, label='Gavel')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Testing Scores')
    ax.set_title('Testing Scores by Configuration and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(configurations)
    ax.legend()

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, cnb_training, width, label='CNB')
    rects2 = ax.bar(x + width/2, gavel_training, width, label='Gavel')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Training Scores')
    ax.set_title('Training Scores by Configuration and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(configurations)
    ax.legend()

    fig.tight_layout()
    plt.show()
