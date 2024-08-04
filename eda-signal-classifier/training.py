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
from utilities.loaders import concur_load_data, load_results
from argparse import ArgumentParser    

def summarize_results(estimator_name: str, results: dict):
    """
    this function will average out each and all folds 
    of each metric value of each hyper param configuration

    args:
        results - a dictionary loaded from the .json file containing
        the results of `tuning.py`
    """

    for hyper_param_config_key, folds_metric_values in results[estimator_name].items():
        # create key out of hyper_param_config
        print(hyper_param_config_key)
        folds_train_acc = np.array(folds_metric_values["folds_train_acc"])
        folds_cross_acc = np.array(folds_metric_values["folds_cross_acc"])
        folds_train_prec = np.array(folds_metric_values["folds_train_prec"])
        folds_cross_prec = np.array(folds_metric_values["folds_cross_prec"])
        folds_train_rec = np.array(folds_metric_values["folds_train_rec"])
        folds_cross_rec = np.array(folds_metric_values["folds_cross_rec"])
        folds_train_f1 = np.array(folds_metric_values["folds_train_f1"])
        folds_cross_f1 = np.array(folds_metric_values["folds_cross_f1"])
        folds_train_roc_auc = np.array(folds_metric_values["folds_train_roc_auc"])
        folds_cross_roc_auc = np.array(folds_metric_values["folds_cross_roc_auc"])

    # train acc cross acc train prec cross prec train rec cross rec

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
    parser.add_argument("-m", type=str, default='lr', help="model e.g. lr for logistic regression, rf for random forest, svm for support vector machine, gbt for gradient boosted tree, that represents what results.json file should be loaded ")
    parser.add_argument("-pl", type=str, default="taylor", 
        help="represents what pipeline which involves what feature set must \
        be kept when data is loaded and what model must the feature selector \
        be based on i.e. SVM or RFC Taylor et al, must be used. (2015) for instance \
        has used most statistical features but variable frequency complex demodulation \
        based features are not used unlike in Hossain et al. (2022) study")
    args = parser.parse_args()

    # read and load data
    subjects_features, subjects_labels, subject_to_id = concur_load_data('./data/Artifact Detection Data/train/', feat_config=args.pl)

    # load results.json 
    results = load_results(estimator_name=args.m)
    summarize_results(estimator_name=args.m, results=results)

    