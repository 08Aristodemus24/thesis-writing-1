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

from sklearn.preprocessing import MinMaxScaler

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
    subjects_inputs, subjects_labels, subjects_name = zip(*subjects_data)

    # convert tuples to list
    subjects_inputs = list(subjects_inputs)
    subjects_labels = list(subjects_labels)
    subjects_names = list(subjects_name)

    # if subjects_inputs and subjects_labels elementes are dataframe
    # objects then it would be better to concatenate them, but if they
    # are np.ndarray's then it would be better to leave them as list
    # elements
    if type(subjects_inputs[0]) == pd.DataFrame:

        subjects_features = pd.concat(subjects_inputs, axis=0, ignore_index=True)
        subjects_labels = pd.concat(subjects_labels, axis=0, ignore_index=True)

        return subjects_features, subjects_labels, subjects_names
    
    return subjects_inputs, subjects_labels, subjects_names



def concur_load_data(feat_config: str="Taylor", data_split: str="train", exc_lof: bool=True):
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

    if feat_config.capitalize() == "Taylor" or feat_config.capitalize() == "Hossain":

        # feature configuration can either be hossain or taylor which will return
        # feature set as a result of reading .txt file containing all features
        # associated to a researcher
        dir = f'./data/Artifact Detection Data/{data_split}/'
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


            return (subject_features, subject_labels, subject_name)

        with ThreadPoolExecutor() as exe:
            # return from this will be a list of all subjects
            # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
            subjects_data = list(exe.map(helper, subject_names))
            subjects_features, subjects_labels, subjects_names = _combine_data(subjects_data)

            # select all features associated with feat_config, include also subject_id
            subjects_features = subjects_features[feat_set + ['subject_id']]

        print("subjects features, labels, names and subject to id lookup loaded")
        return subjects_features, subjects_labels, subjects_names, subject_to_id

    elif feat_config.capitalize() == "Jurado" or feat_config.capitalize() == "Cueva":
        # set directory to find raw subject signals based on data split
        dir = f'./data/Electrodermal Activity artifact correction BEnchmark (EDABE)/{data_split.capitalize()}/'

        # list all .csv features and .csv labels in directory
        subject_names = list(set([re.sub(r".csv", "", file) for file in os.listdir(dir)]))
        subject_to_id = {subject: id for id, subject in enumerate(subject_names)}

        # concurrent processing
        def helper(subject_name: str):
            subject_eda_data = pd.read_csv(f'{dir}{subject_name}.csv', sep=';')
            subject_eda_data.columns = ['time', 'raw_signal', 'clean_signal', 'label', 'auto_signal', 'pred_art', 'post_proc_pred_art']


            subject_signals, subject_labels = charge_raw_data(subject_eda_data, x_col="raw_signal", y_col="label", scale=True)

            return (subject_signals, subject_labels, subject_name)
        
        with ThreadPoolExecutor() as exe:
            # return from this will be a list of all subjects
            # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
            subjects_data = list(exe.map(helper, subject_names))

            # find a way here to combine all 3d subject_signals and subject_labels
            subjects_signals, subjects_labels, subjects_names = _combine_data(subjects_data)

        print("subjects signals, labels, names and subject to id lookup loaded")
        return subjects_signals, subjects_labels, subjects_names, subject_to_id
    
    else:
        # set directory to find raw subject signals based on data split
        dir = f'./data/Hybrid Artifact Detection Data/{data_split}/'

        # list all .csv features and .csv labels in directory
        subject_names = list(set([re.sub(r"_hof.csv|_labels.csv|_lof.csv", "", file) for file in os.listdir(dir)]))
        subject_to_id = {subject: id for id, subject in enumerate(subject_names)}

        # concurrent processing
        def helper(subject_name: str):
            # read higher and lower order features of subject into df
            subject_hof = pd.read_csv(f'{dir}{subject_name}_hof.csv', index_col=0)
            subject_lof = pd.DataFrame() if exc_lof == True else pd.read_csv(f'{dir}{subject_name}_lof.csv', index_col=0)
            

            # concatenate both dataframes of higher and lower features 
            subject_hof_lof = pd.concat([subject_hof, subject_lof], axis=1)

            # after concatenating dataframes containing higher and lower order
            # features of subject assign id column and value to the subject
            subject_hof_lof['subject_id'] = subject_to_id[subject_name]

            subject_labels = pd.read_csv(f'{dir}{subject_name}_labels.csv', index_col=0)
            subject_labels['subject_id'] = subject_to_id[subject_name]

            return (subject_hof_lof, subject_labels, subject_name)
        
        with ThreadPoolExecutor() as exe:
            # return from this will be a list of all subjects
            # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
            subjects_data = list(exe.map(helper, subject_names))

            # find a way here to combine all 3d subject_signals and subject_labels
            subjects_features, subjects_labels, subjects_names = _combine_data(subjects_data)

            # save also feature set of combined hofs and lofs
            # for easier access during training
            feature_set = subjects_features.columns[subjects_features.columns != 'subject_id'].to_list()
            save_lookup_array('./data/Artifact Detection Data/cueva_second_phase_feature_set.txt', feature_set)

        print("subjects features, labels, names and subject to id lookup loaded")
        return subjects_features, subjects_labels, subjects_names, subject_to_id

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
    final_metric_names = None
    
    for hyper_param_config_key, folds_metric_values in results[estimator_name].items():
        mean_metric_values = []
        metric_names = []

        # calculate mean of each folds metric value
        for key, value in folds_metric_values.items():
            mean_metric_value = np.array(value).mean()
            mean_metric_values.append(mean_metric_value)
            metric_names.append(key)

        # mean_train_acc = np.array(folds_metric_values["folds_train_acc"]).mean()
        # mean_cross_acc = np.array(folds_metric_values["folds_cross_acc"]).mean()
        # mean_train_prec = np.array(folds_metric_values["folds_train_prec"]).mean()
        # mean_cross_prec = np.array(folds_metric_values["folds_cross_prec"]).mean()
        # mean_train_rec = np.array(folds_metric_values["folds_train_rec"]).mean()
        # mean_cross_rec = np.array(folds_metric_values["folds_cross_rec"]).mean()
        # mean_train_f1 = np.array(folds_metric_values["folds_train_f1"]).mean()
        # mean_cross_f1 = np.array(folds_metric_values["folds_cross_f1"]).mean()
        # mean_train_roc_auc = np.array(folds_metric_values["folds_train_roc_auc"]).mean()
        # mean_cross_roc_auc = np.array(folds_metric_values["folds_cross_roc_auc"]).mean()
        # mean_train_roc_auc_prob = np.array(folds_metric_values["folds_train_roc_auc_prob"]).mean()
        # mean_cross_roc_auc_prob = np.array(folds_metric_values["folds_cross_roc_auc_prob"]).mean()
        
        # create column
        summarized[hyper_param_config_key] = mean_metric_values
        final_metric_names = metric_names
        
    summarized_results = pd.DataFrame(summarized, index=final_metric_names)
    
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
    and returns this. If no file is found a false boolean is
    returned
    """

    try:
        with open(path, 'r') as file:
            feature_set = file.read()
            feature_set = feature_set.split('\n')
            file.close()

        return feature_set
    except FileNotFoundError as e:
        print("file not found please run needed script first to produce file")
        return False

def save_lookup_array(path: str, uniques: list):
    """
    saves and writes all the unique list of values to a
    a file for later loading by load_lookup_array()
    """
    uniques = [uniques[i] + '\n' for i in range(len(uniques) - 1)] + [uniques[-1]]

    with open(path, 'w') as file:
        file.writelines(uniques)
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


def charge_raw_data(data: pd.DataFrame | np.ndarray, 
    hertz: int=128, 
    window_time: float | int=5, 
    x_col="raw_signal", 
    target_time=0.5, 
    y_col=None, 
    scale=False,
    verbose: bool=False):
    """
    charge_raw_data" preprocesses the input signal cutting the signal in pieces of 5 seconds.
    In the case that a target is introduced i.e. y_col != None, the target is cut the last 0.5
    seconds of the binary target, becoming the target of the correspondent 5 seconds segement.
    """

    # we access the SCR values via raw data column
    x_signals = data[x_col].values

    # here if we would want to create windows of the raw data including the target label
    # we must specify which target label we want to include since there are multiple columns
    # that pertain to the label I vbelieve which are: binary_target, predicted artifacts
    # and post processed artifacts
    if y_col is not None:
        y_signals = data[y_col].values

    window_size = int(window_time * hertz)
    target_size = int(target_time * hertz)

    x_window_list, y_window_list = [], []

    ctr = 0
    # so if we have a length of 765045 rows for the raw eda data
    # and in each row we'd have to multiply 128 to get specific seconds e.g.
    # to get 0th second we multiply 128 by 0 and use it as index
    # raw_eda_df['time'].iloc[:128 * 0], to get 1st second mark we'd have
    # to multiply 128 by 1 and use it as index raw_eda_df['time'].iloc[:128 * 1]

    # but what is the point of subtracting 765045 by window size of 640 (5 * 128)?
    print(f'length of x_signals: {len(x_signals)}')
    print(f'window size: {window_size}')

    signals_len = data.shape[0]
    for i in range(window_size, signals_len, target_size):
        # print(f'start x: {i - window_size} - end x: {i}')
        # iteration pattern is the following
        # 0 <= 765045 - 640 (764405)
        # 64 <= 765045 - 640
        # 128 <= 765045 - 640
        # 192 <= 765045 - 640
        # 256 <= 765045 - 640
        # 320 <= 765045 - 640
        # ...
        # 764288 <= 765045 - 640
        # 764352 <= 765045 - 640 (764405) we only go until here as 764352 + 64
        # (or another 0.5s segment would result in 764416 which is greater than 764405)  

        # maybe what this truly does is we get 5 seconds of a signal and since there are 128 signals per second
        # we would in total get 640 rows for 5 seconds of our signals

        # oh so this is the denominator part of the min max scaling formula
        # and as stated by llanes-jurado et al. they used min max scaling to scale the raw signals
        # mroeover nanmax and min is used in case of nan values in the windows which returns
        # minimum of an array or minimum along an axis, ignoring any NaNs
        # 0:0 + 640 = 0:640
        # 64:64 + 640 = 64:704
        # 128:128 + 640 = 128:768
        # 192:192 + 640 = 192:832
        # ...
        # 764352:764352 + 640 = 764352:764992
        # if we exceed 764352 by adding 64 then we have 764416
        # 764416:764416 + 640 = 764416:765056 and 765056 exceeds the index and rows of 765045 

        # if scale is true then min max scaling is applied
        x_signal = x_signals[(i - window_size):i]
        if scale == True:
            
            denominator_norm = (np.nanmax(x_signal) - np.nanmin(x_signal))
            denominator_norm = denominator_norm + 1e-100 if denominator_norm == 0 else denominator_norm

            # this is full min max scaling formula with the denominator using
            # the difference of the min and max of a window
            # to address also potential zero division concerns
            x_window = (x_signal - np.nanmin(x_signal)) / denominator_norm
        else:
            # this would be appropriate if there was a larger ram
            x_window = x_signal

        # we then append these normed signals to a list
        x_window_list.append(x_window)

        # returns the mean of a list or matrix of values given an
        # axis ignoring any nan values. Based on Llanes-Jurado et al. (2023)
        # the threshold for a 0.5s segment of a signal to be accepted as an
        # artifact must be 0.5 or 50% if it is less than this then the label
        # of such a segment of the signal will be not an artifact
        # 0 + 640 - 64:0 + 640 = 576:640
        # 64 + 640 - 64:64 + 640 = 640:704
        # 128 + 640 - 64:128 + 640 = 704:768
        # ...
        # 764288 + 640 - 64:764288 + 640 = 764864:764928
        # 764352 + 640 - 64:764352 + 640 = 764928:764992
        # this iteration pattern now I know just gets the last 0.5s segment of a 5s segment and 
        
        if y_col is not None:
            y_signal = y_signals[(i - target_size):i]
            cond = np.nanmean(y_signal) > 0.5
            y_window_list.append(1 if cond else 0)
        
        # this will increment our i by the size of our target frames which in this 
        # case is 0.5s or 64 rows since 1 second is 128 rows or 128hz
        ctr += 1
    print(f'number of rows created: {ctr}')

    # because x_window_list and y_window_list when converted to a numpy array will
    # be of dimensions (m, 640) and (m,) respectively we need to first and foremost
    # reshpae x_window_list into a 3D matrix such that it is able to be taken in
    # by an LSTM layer, m being the number of examples, 640 being the number of time steps
    # and 1 being the number of features which will be just our raw eda signals.
    X = np.array(x_window_list)
    subject_signals = np.reshape(X, (X.shape[0], X.shape[1], -1))

    # and because y_window_list is merely of dimension (m, ) we will have to
    # expand its dimensions such that it can be accepted by our tensorflow model
    # resulting shape of subject_labels will now be (m, 1)
    if y_col is not None:
        Y = np.array(y_window_list)
        subject_labels = np.reshape(Y, (Y.shape[0], -1))

        return (subject_signals, subject_labels) 
    
    else:
        return subject_signals

def split_data(subjects_signals: list, subjects_labels: list, test_ratio: int):

    n_subjects = len(subjects_signals)
    train_size = int(n_subjects * (1 - test_ratio))
    
    # split data into training and cross validation
    train_signals = subjects_signals[:train_size]
    train_labels = subjects_labels[:train_size]
    cross_signals = subjects_signals[train_size:]
    cross_labels = subjects_labels[train_size:]

    return train_signals, train_labels, cross_signals, cross_labels


