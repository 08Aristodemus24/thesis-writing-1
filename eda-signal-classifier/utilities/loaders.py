import math
import csv
import numpy as np
import pickle
import json
import os
import pandas as pd
import requests
import zipfile
import re

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.text import tokenizer_from_json


def download_dataset(url):
    response = requests.get("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/w8fxrg4pv5-2.zip", stream=True)
    response.headers

    # extract primary zip file
    with open("./data/EDABE dataset.zip", mode="wb") as file:
        for chunk in response.iter_content(chunk_size=10 * 1024):
            file.write(chunk)

    # extract secondary zip file
    with zipfile.ZipFile('./data/EDABE dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

def _combine_data(subjects_data):
    subjects_features, subjects_labels = zip(*subjects_data)

    subjects_features = pd.concat(subjects_features, axis=0, ignore_index=True)
    subjects_labels = pd.concat(subjects_labels, axis=0, ignore_index=True)

    return subjects_features, subjects_labels

def concur_load_data(dir: str, feat_config: str="Taylor"):
    """
    returns the features, labels, and subject ids

    args:
        feat_set - represents what feature set must be
        kept when data is loaded. Taylor et al. (2015)
        for instance has used most statistical features
        but variable frequency complex demodulation based
        features are not used unlike in Hossain et al.
        (2022) study
    """

    # feature configuration can either be hossain or taylor which will return
    # feature set as a result of reading .txt file containing all features
    # associated to a researcher
    feat_set = np.genfromtxt(f'./data/Artifact Detection Data/{feat_config.lower()}_feature_set.txt', dtype=str).tolist() 

    # list all .csv features and .csv labels in directory
    subject_names = list(set([re.sub(r"_features.csv|_labels.csv", "", file) for file in os.listdir(dir)]))
    subject_to_id = {subject: id for id, subject in enumerate(subject_names)}

    # read all .csv features and .csv labels
    def helper(subject_name: str):
        # read and assign id to the subject 
        subject_features = pd.read_csv(f'{dir}{subject_name}_features.csv', index_col=0)
        subject_features['subject_id'] = subject_to_id[subject_name]

        subject_labels = pd.read_csv(f'{dir}{subject_name}_labels.csv', index_col=0)
        subject_labels['subject_id'] = subject_to_id[subject_name]


        return (subject_features, subject_labels)

    with ThreadPoolExecutor() as exe:
        # return from this will be a list of all subjects
        # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
        subjects_data = list(exe.map(helper, subject_names))
        subjects_features, subjects_labels = _combine_data(subjects_data)

        # select all features associated with feat_config, include also subject_id
        subjects_features = subjects_features[feat_set + ['subject_id']]

    return subjects_features, subjects_labels, subject_to_id

def load_results(filename: str):
    # read .json file and maybe convert it to a readable dataframe
    # that you can decide which hyper params give the highest mean cross metric value
    # if json file exists read and use it
    if os.path.exists(f'./results/{filename}'):
        # read json file as dictionary
        with open(f'./results/{filename}') as file:
            results = json.load(file)
            file.close()

        return results

    else:
        raise FileNotFoundError("File not found please run `tuning.py` first to obtain `.json` file of results!")

def summarize_results(estimator_name: str, results: dict):
    """
    this function will average out each and all folds 
    of each metric value of each hyper param configuration

    args:
        results - a dictionary loaded from the .json file containing
        the results of `tuning.py`

        e.g. 
        {
            "gbt": {
                "C_1": {
                    "folds_train_acc": [0.56, 0.75, 0.80, 0.66],
                    "folds_cross_acc": [0.66, 0.85, 0.79, 0.76],
                    "folds_train_roc_auc": [0.76, 0.78, 0.80, 0.74],
                    "folds_cross_roc_auc": [0.70, 0.77, 0.82, 0.66]
                },
                "C_10": {
                    "folds_train_acc": [0.86, 0.75, 0.81, 0.79],
                    "folds_cross_acc": [0.78, 0.81, 0.80, 0.80],
                    "folds_train_roc_auc": [0.79, 0.85, 0.90, 0.86],
                    "folds_cross_roc_auc": [0.89, 0.79, 0.80, 0.76]
                },
                "C_100": {
                    "folds_train_acc": [0.89, 0.86, 0.84, 0.91],
                    "folds_cross_acc": [0.79, 0.75, 0.90, 0.86],
                    "folds_train_roc_auc": [0.89, 0.79, 0.80, 0.76],
                    "folds_cross_roc_auc": [0.90, 0.74, 0.83, 0.86]
                },
                "C_1000": {
                    "folds_train_acc": [0.78, 0.81, 0.80, 0.80],
                    "folds_cross_acc": [0.86, 0.90, 0.84, 0.76],
                    "folds_train_roc_auc": [0.89, 0.86, 0.84, 0.91],
                    "folds_cross_roc_auc": [0.90, 0.75, 0.81, 0.85]
                }
            }
        }

    returns a dataframe which looks like this:
                       | hyper param config 1 | hyper param config 2 | hyper param config 3 |
    mean train acc     |                      |                      |                      |
    mean cross acc     |                      |                      |                      |
    mean train roc auc |                      |                      |                      |
    mean cross roc auc |                      |                      |                      |
    """

    summarized = {}
    for hyper_param_config_key, folds_metric_values in results[estimator_name].items():

        # calculate mean of each folds metric value
        mean_train_acc = np.array(folds_metric_values["folds_train_acc"]).mean()
        mean_cross_acc = np.array(folds_metric_values["folds_cross_acc"]).mean()
        mean_train_prec = np.array(folds_metric_values["folds_train_prec"]).mean()
        mean_cross_prec = np.array(folds_metric_values["folds_cross_prec"]).mean()
        mean_train_rec = np.array(folds_metric_values["folds_train_rec"]).mean()
        mean_cross_rec = np.array(folds_metric_values["folds_cross_rec"]).mean()
        mean_train_f1 = np.array(folds_metric_values["folds_train_f1"]).mean()
        mean_cross_f1 = np.array(folds_metric_values["folds_cross_f1"]).mean()
        mean_train_roc_auc = np.array(folds_metric_values["folds_train_roc_auc"]).mean()
        mean_cross_roc_auc = np.array(folds_metric_values["folds_cross_roc_auc"]).mean()
        
        # create column
        summarized[hyper_param_config_key] = [
            mean_train_acc,
            mean_cross_acc,
            mean_train_prec,
            mean_cross_prec,
            mean_train_rec,
            mean_cross_rec,
            mean_train_f1,
            mean_cross_f1,
            mean_train_roc_auc,
            mean_cross_roc_auc]
        
    summarized_results = pd.DataFrame(summarized, index=[
        "mean_train_acc",
        "mean_cross_acc",
        "mean_train_prec",
        "mean_cross_prec",
        "mean_train_rec",
        "mean_cross_rec",
        "mean_train_f1",
        "mean_cross_f1",
        "mean_train_roc_auc",
        "mean_cross_roc_auc"])
    
    return summarized_results

# def device_exists():
#     """
#     returns true if gpu device exists
#     """

#     device_name = tf.test.gpu_device_name()
#     if device_name != '/device:GPU:0':
#         return False
#     return True

def load_lookup_array(path: str):
    """
    reads a text file containing a list of all unique values
    and returns this
    """

    with open(path, 'rb') as file:
        char_to_idx = pickle.load(file)
        file.close()

    return char_to_idx

def save_lookup_array(path: str, uniques: list):
    """
    saves and writes all the unique list of values to a
    a file for later loading by load_lookup_array()
    """

    with open(path, 'wb') as file:
        pickle.dump(uniques, file)
        file.close()

def save_meta_data(path: str, meta_data: dict):
    """
    saves dictionary of meta data such as hyper 
    parameters to a .json file
    """

    with open(path, 'w') as file:
        json.dump(meta_data, file)
        file.close()

def load_meta_data(path: str):
    """
    loads the saved dictionary of meta data such as
    hyper parameters from the created .json file
    """

    with open(path, 'r') as file:
        meta_data = json.load(file)
        file.close()

    return meta_data

def save_model(model, path: str):
    """
    saves partcularly an sklearn model in a .pkl file
    for later testing
    """

    with open(path, 'wb') as file:
        pickle.dump(model, file)
        file.close()

def load_model(path: str):
    """
    loads the sklearn model, scaler, or encoder stored
    in a .pkl file for later testing and deployment
    """

    with open(path, 'rb') as file:
        model = pickle.load(file)
        file.close()

    return model

def create_metrics_df(train_metric_values, 
                      val_metric_values, 
                      test_metric_values, 
                      metrics=['accuracy', 'precision', 'recall', 'f1-score']):
    """
    creates a metrics dataframe
    """

    metrics_dict = {
        'data_split': ['training', 'validation', 'testing']
    }

    for index, metric in enumerate(metrics):
        metrics_dict[metric] = [
            train_metric_values[index], 
            val_metric_values[index],
            test_metric_values[index]
        ]

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df

def create_classified_df(train_conf_matrix, val_conf_matrix, test_conf_matrix, train_labels, val_labels, test_labels):
    """
    creates a dataframe that represents all classified and 
    misclassified values
    """

    num_right_cm_train = train_conf_matrix.trace()
    num_right_cm_val = val_conf_matrix.trace()
    num_right_cm_test = test_conf_matrix.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    
    return classified_df


def charge_raw_data(df, x_col="rawdata", target_size_frames=64, y_col=None, freq_signal=128, verbose=False):
    """
    charge_raw_data" preprocesses the input signal cutting the signal in pieces of 5 seconds. 
    In the case that a target is introduced i.e. y_col != None, the target is cut the last 0.5 
    seconds of the binary target, becoming the target of the correspondent 5 seconds segement.
    """
    
    # we access the SCR values via raw data column
    x_signal = df[x_col].values

    # here if we would want to create windows of the raw data including the target label
    # we must specify which target label we want to include since there are multiple columns
    # that pertain to the label I vbelieve which are: binary_target, predicted artifacts 
    # and post processed artifacts
    if y_col is not None:
        y_signal = df[y_col].values
    
    window_size = 5 * freq_signal

    x_window_list, y_window_list = [], []
    
    i = 0
    # so if we have a length of 765045 rows for the raw eda data
    # and in each row we'd have to multiply 128 to get specific seconds e.g.
    # to get 0th second we multiply 128 by 0 and use it as index 
    # raw_eda_df['time'].iloc[:128 * 0], to get 1st second mark we'd have
    # to multiply 128 by 1 and use it as index raw_eda_df['time'].iloc[:128 * 1]

    # but what is the point of subtracting 765045 by window size of 640 (5 * 128)?
    print(f'length of x_signals: {len(x_signal)}')
    print(f'window size: {window_size}')
    while i <= len(x_signal) - window_size:
        
        # oh so this is the denominator part of the min max scaling formula
        # and as stated by llanes-jurado et al. they used min max scaling to scale the raw signals
        # mroeover nanmax and min is used in case of nan values in the windows which returns 
        # minimum of an array or minimum along an axis, ignoring any NaNs
        denominator_norm = (np.nanmax(x_signal[i:(i + window_size)]) - np.nanmin(x_signal[i:(i + window_size)]))
        
        # this is full min max scaling formula with the denominator using 
        # the difference of the min and max of a window
        x_signal_norm = (x_signal[i:(i + window_size)] - np.nanmin(x_signal[i:(i + window_size)])) / denominator_norm

        # we then append these normed signals to a list
        x_window_list.append(x_signal_norm)
        
        if y_col is not None:
            # returns the mean of a list or matrix of values given an axis ignoring 
            # any nan values. Here according to Llanes-Jurado et al. (2023)'s paper 
            # if more than 50% of the segment was labeled as an artifact, such a
            # segment of 0.5 s was labeled indeed as an artifact
            cond = np.nanmean(y_signal[(i + window_size - target_size_frames):(i + window_size)]) > 0.5
            y_window_list.append(1 if cond else 0)

        if i % 50000 == 0 and verbose:
            print("Iteration", i, "of", len(x_signal) - window_size - 1, end="\r")
        
        print(i)
        i += target_size_frames
    
    return np.array(x_window_list), np.array(y_window_list)