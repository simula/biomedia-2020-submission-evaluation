#!/usr/bin/env python

import os
import csv
import random
import warnings

from sklearn import metrics
from itertools import product
from shutil import copyfile

import numpy as np

random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

SUBMISSIONS_DIRECTORY_PATH = ""

GROUND_TRUTH_PATH = ""
RESULTS_DIRECTORY = ""

def read_submission(submission_path):
    results = {}
    duplicates = {}
    lines = []
    with open(submission_path) as f:
        for row in csv.reader(f):
            lines.append(row)
            image_id = os.path.splitext(row[0].strip())[0].strip()

            if image_id in results:
                duplicates[image_id] = row[1].strip()
                continue
                
            results[image_id] = row[1].strip()
    return duplicates, np.array(lines), results

def read_csv(gt_path, variables):

    ground_truth = { }

    with open(gt_path) as csv_data:

        csv_reader = csv.reader(csv_data, delimiter=";")
        header = next(csv_reader)
        variable_indecies = [header.index(variable) for variable in variables]

        for row in csv_reader:
            video_id = os.path.splitext(row[0].strip())[0].strip()                
            ground_truth[ video_id ] = [ float(row[ variable_index ]) for variable_index in variable_indecies ]
            
    return ground_truth

def evaluate_submission(submission_path):

    submission_attributes = os.path.basename(submission_filename).split("_")
    
    team_name = submission_attributes[1]
    task_name = submission_attributes[2]
    run_id    = os.path.splitext("_".join(submission_attributes[3:]))[0]

    if task_name == "motility":
        variables = [ "progressive_%", "non_progressive_%", "immotile_%" ]
    elif task_name == "morphology":
        variables = [ "head_defect_%", "midpiece_defect_%", "tail_defect_%" ]
    else:
        raise Exception("Error!")

    team_result_path = os.path.join(RESULTS_DIRECTORY, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)

    gt_results = read_csv(GROUND_TRUTH_PATH, variables)
    pred_results = read_csv(submission_path, variables)

    y_pred, y_truth = [], []

    for video_id, actual_class in gt_results.items():
        y_pred.append(gt_results[ video_id ])
        y_truth.append(pred_results[ video_id ])

    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)

    if len(y_truth) != len(y_pred):
        raise Exception("The number of predicted values is NOT equal to the ground truth!")

    mean_absolute_error = metrics.mean_absolute_error(y_truth, y_pred, multioutput="raw_values")
    mean_squared_error = metrics.mean_squared_error(y_truth, y_pred, multioutput="raw_values")
    root_mean_squared_error = metrics.mean_squared_error(y_truth, y_pred, multioutput="raw_values", squared=False)
    mean_squared_log_error = metrics.mean_squared_log_error(y_truth, y_pred, multioutput="raw_values")
    median_absolute_error = metrics.median_absolute_error(y_truth, y_pred, multioutput="raw_values")

    with open(os.path.join(team_result_path, "official_biomedia_2020_%s_%s_%s_metrics.txt" % (team_name, task_name, run_id)), "w") as f:

        for index, variable_name in enumerate(variables):
            
            f.write("Results for %s\n" % variable_name)
            f.write("max error: %s\n" % metrics.max_error(y_truth[:, index], y_pred[:, index]))
            f.write("mean absolute error: %s\n" % mean_absolute_error[index])
            f.write("mean squared error: %s\n" % mean_squared_error[index])
            f.write("root mean squared error: %s\n" % root_mean_squared_error[index])
            f.write("mean squared log error: %s\n" % mean_squared_log_error[index])
            f.write("median absolute error: %s\n" % median_absolute_error[index])
            f.write("\n")

        f.write("Average results\n")
        f.write("mean absolute error: %s\n" % np.mean(mean_absolute_error))
        f.write("mean squared error: %s\n" % np.mean(mean_squared_error))
        f.write("root mean squared error: %s\n" % np.mean(root_mean_squared_error))
        f.write("mean squared log error: %s\n" % np.mean(mean_squared_log_error))
        f.write("median absolute error: %s\n" % np.mean(median_absolute_error))
        
if __name__ == "__main__":

    for submission_filename in os.listdir(SUBMISSIONS_DIRECTORY_PATH):

        if not os.path.splitext(submission_filename)[1] == ".csv":
            continue

        print(f"Evaluating { submission_filename }...")

        evaluate_submission(os.path.join(SUBMISSIONS_DIRECTORY_PATH, submission_filename))