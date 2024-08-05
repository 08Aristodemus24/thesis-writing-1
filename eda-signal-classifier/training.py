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

from utilities.loaders import concur_load_data
from argparse import ArgumentParser    




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

    