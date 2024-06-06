
from predictor import train, predict
from dataLoader import read_data
import time
import os
import pandas as pd


dataset = "code_only"
path_results = "gh_action_dataset/results_5_fold_"+dataset+"_multi"
results_file = "results_complete_" + dataset

dataset = "gh_action_dataset/" + dataset + ".csv"

def remove_comments_line(list):
    res = []
    for e in list:
        if e.find('#') == -1:
            res.append(e)

    return res


def map_testing_project(testing,project):
    projs = []
    for key_test in testing.keys():
        projs.append(project.get(key_test))

    return projs




def run_cnb():
    for cutoff in range(1, 3):
        succ_scores = []
        pr_scores = []
        rec_scores = []
        f1_scores = []
        train_tot = 0
        testing_time = 0
        for i in range(1, 6):
            out_path = path_results + '/cutoff_' + str(cutoff) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            train_data, test_data, test_projects = read_data(dataset)
            out_file = out_path + results_file + '_round_' + str(i) + ".csv"

            results = []


            start_training = time.time()
            model, tf_idf, vects = train(train_data)
            end_training = time.time()

            train_tot += end_training-start_training
            print("training time", end_training-start_training)
            for desc, actual_topics in test_data.items():
                if len(actual_topics) > 0:
                    start_testing = time.time()
                    predicted_topics = predict(desc, model, tf_idf, vects)
                    end_testing = time.time()
                    proj_id = test_projects.get(desc)
                    testing_time += end_testing - start_testing
                    print("testing time", testing_time)
                    string_pred = ','.join(predicted_topics[:cutoff])
                    string_actual = ','.join(actual_topics)

                    #res.write(f'{proj_id},{string_actual},{string_pred}\n')
                    results.append({
                        "project": proj_id,
                        "actual": string_actual,
                        "predicted": string_pred
                    })

                # Convert results to a DataFrame
                df = pd.DataFrame(results)

                # Write the DataFrame to a CSV file with the header
                try:
                    df.to_csv(out_file, index=False, encoding='utf-8', errors='ignore', header=["project", "actual", "predicted"])
                except:
                    print("error in", out_file)
                    continue

        print("avg testing", testing_time/5)
        print("avg training", train_tot / 5)




if __name__ == '__main__':
    run_cnb()






