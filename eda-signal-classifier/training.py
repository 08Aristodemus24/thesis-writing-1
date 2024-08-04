import os
import pandas as pd
import re
import itertools
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# import tensorflow as tf
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tuning import concur_load_data
from argparse import ArgumentParser

def _combine_data(subjects_data):
    """
    args:
        subjects_data - is a list of dataframes representing each
        subjects calculated features
    """
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
        """ this can't be since I cant coerce a column that are float values to strictly int zeros"""
        subject_features['subject_id'] = subject_to_id[subject_name]
        

        subject_labels = pd.read_csv(f'{dir}{subject_name}_labels.csv', index_col=0)
        subject_labels['subject_id'] = subject_to_id[subject_name]


        return (subject_features, subject_labels)

    with ThreadPoolExecutor() as exe:
        # return from this will be a list of all subjects
        # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
        subjects_data = list(exe.map(helper, subject_names))
        subjects_features, subjects_labels = _combine_data(subjects_data)
        subjects_features = subjects_features[feat_set]

    return subjects_features, subjects_labels, subject_to_id



def load_results(estimator_name):
    # read .json file and maybe convert it to a readable dataframe
    # that you can decide which hyper params give the highest mean cross metric value
    # if json file exists read and use it
    if os.path.exists(f'./results/{estimator_name}_results.json'):
        # read json file as dictionary
        with open(f'./results/{estimator_name}_results.json') as file:
            results = json.loads(file)
            file.close()

        return results

    else:
        raise FileNotFoundError("File not found please run `training.py` first to obtain `.json` file of results!")
    

def summarize_results(results: dict):
    """
    this function will average out each and all folds 
    of each metric value of each hyper param configuration

    args:
        results - a dictionary loaded from the .json file containing
        the results of `tuning.py`
    """




def loso_cross_validation(subjects_features: pd.DataFrame, subjects_labels: pd.DataFrame, subject_to_id: dict, estimator_name, estimator, **hyper_param_config: dict):
    """
    args:
        subjects_features: pd.DataFrame - 
        subjects_labels: pd.DataFrame - 
        subject_to_id: dict - 
        model - 
        hyper_param_config: dict - 
    """

    # create key out of hyper_param_config
    hyper_param_config_key = "|".join([f"{hyper_param}_{value}" for hyper_param, value in hyper_param_config.items()])



if __name__ == "__main__":
    """
    1. read the .json file containing all performance metric values of each hyper param configuration across all folds of the data of the model
    2. use the feature set used in cross validating all the models
    3. identify hyperparameter configuration with highest mean cross metric value. This will be done by
        - reading the .json file containing all performance metric values and doing some calculations to find out which
        hyper param config does give the best mean cross metric value
    4. retrain a model on the feature set and on these hyper param config on the whole training and validation set we used for tuning but now combined
        - why this is is because we couldn't have really saved a model on a particular fold since it would just have been trained on an incomplete set of
        data. Not including the supposed to be validation set, which makes our model more biased on the training data and not on the validation data
        - Another reason is if indeed a specific hyper param config produced the highest mean cross metric value how would we then save a model when in
        reality there would be multiple models trained on different folds? You may think we can combine them but, we actually can't in reality because 
        there is no means/method in doing so, so the best way is to instead retrain a model on a specific set of hyper param config again that produced
        the highest mean cross metric value during cross validation
    """

    # read and parse user arguments
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, default='lr', help="model e.g. lr for logistic regression, rf for random forest, svm for support vector machine, gbt for gradient boosted tree, to train and validate ")
    parser.add_argument("--feature_config", type=str, default="taylor", 
        help="represents what feature set must be kept when data is loaded. \
        Taylor et al. (2015) for instance has used most statistical features \
        but variable frequency complex demodulation based features are not \
        used unlike in Hossain et al. (2022) study")
    args = parser.parse_args()

    # read and load data
    subjects_features, subjects_labels, subject_to_id = concur_load_data('./data/Artifact Detection Data/train/', feat_config=args.feature_config)



    