from sklearnex.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.svm import SVC, LinearSVC

import itertools
import json
import os
import pandas as pd
import numpy as np
import ast
import re

from utilities.loaders import concur_load_data, save_meta_data, split_data, save_model

from argparse import ArgumentParser



def sample_ids(subjects_lstm_features, n_rows_to_sample):
    """
    args:
        n_rows_to_sample - 
    """

    n_rows_to_sample = n_rows_to_sample if n_rows_to_sample != None else subjects_lstm_features.shape[0]
    sampled_ids = np.random.choice(subjects_lstm_features.shape[0], size=n_rows_to_sample)

    return sampled_ids

def leave_one_subject_out(subjects_lstm_features: list[np.ndarray], subjects_lstm_labels: list[np.ndarray], subject_id: int):
    """
    args:
        lstm_features - 
        labels - 
        subject_id - id of the subject to leave out from the set 
        of subjects features and set of subjects labels
    """

    # make copy first of list then use it to pop, so as to not
    # mutate original list, pop the element with the index that
    # matches the subject id
    copied_lstm_features = subjects_lstm_features[:]
    copied_lstm_labels = subjects_lstm_labels[:]
    cross_lstm_features = copied_lstm_features.pop(subject_id)
    cross_lstm_labels = copied_lstm_labels.pop(subject_id).ravel()

    # after popping the copied lstm_features and labels would have now
    # turned into our train data and our popped elements from our copied
    # lstm_features and labels would now be our cross validation data, and because
    # unlike our cross_lstm_features and cross_labels directly being np.ndarrays
    # now after popping, we have to concatenate the list of our copied lstm_features
    # and labels which is our train data across the 0th dimension since the 
    # dimension of each element is (m, window_size, 1)
    train_lstm_features = np.concatenate(copied_lstm_features, axis=0)
    train_lstm_labels = np.concatenate(copied_lstm_labels, axis=0).ravel()

    return train_lstm_features, train_lstm_labels, cross_lstm_features, cross_lstm_labels

def check_file_key(selector_config, estimator_name, hyper_param_config_key):
    # if json file exists read and use it
    if os.path.exists(f'./results/{selector_config}_{estimator_name}_results.json'):
        # read json file as dictionary
        with open(f'./results/{selector_config}_{estimator_name}_results.json') as file:
            results = json.load(file)
        
            # also if hyper param already exists as a key then 
            # move on to next hyper param by returning from function
            if hyper_param_config_key in results[estimator_name]:
                return False
            
            file.close()
        return results

    # if json file does not exist create one and read as dictionary
    else:
        # will be populated later during loso cross validation
        results = {
            f'{estimator_name}': {
                # 'hyper_param_config_1': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },

                # 'hyper_param_config_2': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },

                # 'hyper_param_config_n': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },
            }
        }

        return results

def loso_cross_validation(subjects_lstm_features: pd.DataFrame | np.ndarray,
    subjects_lstm_labels: pd.DataFrame | np.ndarray,
    subject_to_id: dict,
    selector_config: str,
    estimator_name,
    estimator,
    hyper_param_config: dict):
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

    # if file exists or not return a dictionary but if hyper param 
    # config key already exists return from function
    if check_file_key(selector_config, estimator_name, hyper_param_config_key) != False:
        results = check_file_key(selector_config, estimator_name, hyper_param_config_key)
    else:
        return
    
    # create model with specific hyper param configurations
    model = estimator(**hyper_param_config, verbose=1)

    # initialize empty lists to collect all metric values per fold
    folds_train_acc = []
    folds_train_prec = []
    folds_train_rec = []
    folds_train_f1 = []
    folds_train_roc_auc = []
    folds_cross_acc = []
    folds_cross_prec = []
    folds_cross_rec = []
    folds_cross_f1 = []
    folds_cross_roc_auc = []
    folds_train_roc_auc_prob = []
    folds_cross_roc_auc_prob = []

    # split features and labels into train and cross by 
    # leaving 1 subject out for cross validatoin and the
    # rest for training, iterated for all subjects
    for subject_id in subject_to_id.values():
        # split data by leaving one subject out for testing
        # and the rest for training

        # subjects_features will initially one giant dataframe of 
        # all subjects features extracted from their signals, we 
        # want to at every fold have the train features scaled and 
        # normalized and the cross features scaled and normalized 
        # using the scaler fitted on the train features 
        train_lstm_features, train_lstm_labels, cross_lstm_features, cross_lstm_labels = leave_one_subject_out(subjects_lstm_features, subjects_lstm_labels, subject_id)

        # train model
        print('commencing tuning...')
        model.fit(train_lstm_features, train_lstm_labels)

        # compare true cross and train labels to pred cross and train labels
        pred_train_labels = model.predict(train_lstm_features)
        pred_cross_labels = model.predict(cross_lstm_features)
        pred_train_probs = model.predict_proba(train_lstm_features)
        pred_cross_probs = model.predict_proba(cross_lstm_features)

        # compute performance metric values for each fold
    #     fold_train_acc = accuracy_score(y_true=train_lstm_labels, y_pred=pred_train_labels)
    #     fold_cross_acc = accuracy_score(y_true=cross_lstm_labels, y_pred=pred_cross_labels)
    #     fold_train_prec = precision_score(y_true=train_lstm_labels, y_pred=pred_train_labels)
    #     fold_cross_prec = precision_score(y_true=cross_lstm_labels, y_pred=pred_cross_labels)
    #     fold_train_rec = recall_score(y_true=train_lstm_labels, y_pred=pred_train_labels)
    #     fold_cross_rec = recall_score(y_true=cross_lstm_labels, y_pred=pred_cross_labels)
    #     fold_train_f1 = f1_score(y_true=train_lstm_labels, y_pred=pred_train_labels)
    #     fold_cross_f1 = f1_score(y_true=cross_lstm_labels, y_pred=pred_cross_labels)
    #     fold_train_roc_auc = roc_auc_score(y_true=train_lstm_labels, y_score=pred_train_labels)
    #     fold_cross_roc_auc = roc_auc_score(y_true=cross_lstm_labels, y_score=pred_cross_labels)
    #     fold_train_roc_auc_prob = roc_auc_score(y_true=train_lstm_labels, y_score=pred_train_probs[:, 1])
    #     fold_cross_roc_auc_prob = roc_auc_score(y_true=cross_lstm_labels, y_score=pred_cross_probs[:, 1])
        
    #     # save append each metric value to each respective list
    #     folds_train_acc.append(fold_train_acc)
    #     folds_cross_acc.append(fold_cross_acc)
    #     folds_train_prec.append(fold_train_prec)
    #     folds_cross_prec.append(fold_cross_prec)
    #     folds_train_rec.append(fold_train_rec)
    #     folds_cross_rec.append(fold_cross_rec)
    #     folds_train_f1.append(fold_train_f1)
    #     folds_cross_f1.append(fold_cross_f1)
    #     folds_train_roc_auc.append(fold_train_roc_auc)
    #     folds_cross_roc_auc.append(fold_cross_roc_auc)
    #     folds_train_roc_auc_prob.append(fold_train_roc_auc_prob)
    #     folds_cross_roc_auc_prob.append(fold_cross_roc_auc_prob)

    #     print(f"fold: {subject_id} with hyper params: {hyper_param_config} \
    #           \ntrain acc: {fold_train_acc} cross acc: {fold_cross_acc} \
    #           \ntrain prec: {fold_train_prec} cross prec: {fold_cross_prec} \
    #           \ntrain rec: {fold_train_rec} cross rec: {fold_cross_rec} \
    #           \ntrain f1: {fold_train_f1} cross f1: {fold_cross_f1} \
    #           \ntrain roc_auc: {fold_train_roc_auc} cross roc_auc: {fold_cross_roc_auc} \
    #           \ntrain roc_auc_prob: {fold_train_roc_auc_prob} cross roc_auc_prob: {fold_cross_roc_auc_prob}")

    # # once all fold train and cross metric values collected update read
    # # dictionary with specific hyper param config as key and its recorded
    # # metric values as value
    # results[f'{estimator_name}'][hyper_param_config_key] = {
    #     'folds_train_acc':  folds_train_acc,
    #     'folds_cross_acc': folds_cross_acc,
    #     'folds_train_prec': folds_train_prec,
    #     'folds_cross_prec': folds_cross_prec,
    #     'folds_train_rec': folds_train_rec,
    #     'folds_cross_rec': folds_cross_rec,
    #     'folds_train_f1': folds_train_f1,
    #     'folds_cross_f1': folds_cross_f1,
    #     'folds_train_roc_auc': folds_train_roc_auc,
    #     'folds_cross_roc_auc': folds_cross_roc_auc,
    #     'folds_train_roc_auc_prob': folds_train_roc_auc_prob,
    #     'folds_cross_roc_auc_prob': folds_cross_roc_auc_prob
    # }

    # with open(f'results/{selector_config}_{estimator_name}_results.json', 'w') as file:
    #     json.dump(results, file)

def grid_search_loso_cv(subjects_lstm_features: pd.DataFrame | np.ndarray,
    subjects_lstm_labels: pd.DataFrame | np.ndarray,
    subject_to_id: dict,
    selector_config: str,
    n_rows_to_sample: int | None,
    estimator_name: str,
    estimator,
    hyper_params: dict,):
    """
    args:
        hyper_params - is a dictionary containing all the hyperparameters
        to be used in the model and the respective values to try

        e.g. >>> hyper_params = {'n_estimators': [10, 50, 100], 'max_depth': [3], 'gamma': [1, 10, 100, 1000]}
        >>> list(itertools.product(*list(hyper_params.values())))
        [(10, 3, 1), (10, 3, 10), (10, 3, 100), (10, 3, 1000), (50, 3, 1), (50, 3, 10), (50, 3, 100), (50, 3, 1000), (100, 3, 1), (100, 3, 10), (100, 3, 100), (100, 3, 1000)]
        >>>
        >>> keys, values = zip(*hyper_params.items())
        >>> perm_dicts = [dict(zip(keys, prod)) for prod in itertools.product(*values)]
        >>> perm_dicts
        [{'n_estimators': 10, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 1000}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 1000}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 1000}]
        >>>

        note in the passing of hyper param config dictionary to a function we can always unpack it by:
        >>> dict = {'a': 1, 'b': 2}
        >>> def myFunc(a=0, b=0, c=0):
        >>>     print(a, b, c)
        >>>
        >>> myFunc(**dict)
    """

    # unpack the dictionaries items and separate into list of keys and values
    # ('n_estimators', 'max_depth', 'gamma'), ([10, 50, 100], [3], [1, 10, 100, 1000])
    keys, values = zip(*hyper_params.items())

    # since values is an iterable and product receives variable arguments,
    # variable arg in product are essentially split thus above values
    # will be split into [10, 50, 100], [3], and [1, 10, 100, 1000]
    # this can be then easily used to produce all possible permutations of
    # these values e.g. (10, 3, 1), (10, 3, 10), (10, 3, 100) and so on...
    for prod in itertools.product(*values):
        # we use the possible permutations and create a dictionary
        # of the same keys as hyper params
        hyper_param_config = dict(zip(keys, prod))
        loso_cross_validation(
            subjects_lstm_features, 
            subjects_lstm_labels, 
            subject_to_id, 
            selector_config,
            estimator_name,
            estimator,
            hyper_param_config)



def train_final_estimator(subjects_lstm_features: pd.DataFrame | np.ndarray,
    subjects_lstm_labels: pd.DataFrame | np.ndarray,
    selector_config: str,
    estimator_name: str,
    estimator,
    hyper_param_config: dict):

    # assign to x and y vars for readability
    X = subjects_lstm_features
    Y = subjects_lstm_labels.ravel()

    # create model with specific hyper param configurations
    # and fit to whole training and validation dataset
    model = estimator(**hyper_param_config, verbose=1)
    model.fit(X, Y)
    score = model.score(X, Y)
    print(f'{estimator_name} score: {score}')

    save_model(model, f'./saved/models/{selector_config}_{estimator_name}_clf.pkl')

def create_hyper_param_config(hyper_param_list: list[str]):
    """
    will create a hyper param config dicitonary containing the specific
    values of each hyper param for a model to train with
    args:
        hyper_param_list - is a list of strings containing 
        the hyper param name and its respective values
        that will be parsed and extracted its key
        and value pairs to return as a dictionary
    """

    hyper_param_config = {}
    hyper_param_pattern = r'[A-Za-z_]+'
    value_pattern = r'\d+(\.\d+)?'
    
    for hyper_param in hyper_param_list:
        # extract hyper param name and strip its last occuring underscore
        key = re.search(hyper_param_pattern, hyper_param)[0].strip('_')

        # extract hyper param value and convert to 
        # appropriate data type using literal evaluator
        value = re.search(value_pattern, hyper_param)[0]
        hyper_param_config[key] = ast.literal_eval(value)
    
    return hyper_param_config



if __name__ == "__main__":
    # read and parse user arguments
    parser = ArgumentParser()
    parser.add_argument("--n_features_to_select", type=int, default=40, help="number of features to select by RFE")
    parser.add_argument("--n_rows_to_sample", type=int, default=None, help="number of rows to sample during feature selection by RFE")
    parser.add_argument("-m", type=str, default='lr', help="model e.g. lr for logistic regression, rf for random forest, svm for support vector machine, gbt for gradient boosted tree, to train and validate ")
    parser.add_argument("-pl", type=str, default="taylor", 
        help="represents what pipeline which involves what feature set must \
        be kept when data is loaded and what model must the feature selector \
        be based on i.e. SVM or RFC Taylor et al. (2015), must be used. \
        Taylor et al. (2015) for instance has used most statistical features but \
        variable frequency complex demodulation based features are not used unlike \
        in Hossain et al. (2022) study")
    parser.add_argument("--mode", type=str, default="tuning", help="tuning mode will not \
        save model/s during fitting, while training mode saves model single model with \
        specific hyper param config")
    parser.add_argument("--hyper_param_list", type=str, default="C_100", nargs="+", help="list of hyper parameters to be used as configuration during training")
    args = parser.parse_args()

    # read and load data
    print(os.getcwd())
    subjects_lstm_features, subjects_lstm_labels, subjects_names, subject_to_id = concur_load_data(feat_config=args.pl)
    print(subjects_lstm_features)
    print(subjects_lstm_labels)

    # # create and load test data
    # m = 1000
    # n_f = 32
    # n_subjects = 5
    # subjects_lstm_features = [np.random.randn(m, n_f) for _ in range(n_subjects)]
    # subjects_lstm_labels = [np.random.randint(low=0, high=2, size=(m, 1)) for _ in range(n_subjects)]
    # subject_to_id = {subject: id for id, subject in enumerate(range(n_subjects))}

    # model hyper params
    models = {
        'svm': {
            'model': SVC,
            'hyper_params': {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1], 'probability': [True]}
        }
    }

    if args.mode.lower() == "tuning":
        
        # do feature selection, hyperparameter tuning, 
        # loso cross validation across all subjects, and
        # save model & results
        grid_search_loso_cv(
            subjects_lstm_features, 
            subjects_lstm_labels, 
            subject_to_id, 
            selector_config=args.pl,
            n_rows_to_sample=args.n_rows_to_sample,
            estimator_name=args.m,
            estimator=models[args.m]['model'],
            hyper_params=models[args.m]['hyper_params'],
        )

    elif args.mode.lower() == "training":
        # build hyper param config dictionary from input
        hyper_param_config = create_hyper_param_config(hyper_param_list=args.hyper_param_list)
        
        # we can just modify this script such that it doesn't loop through hyper param configs anymore and
        # will just only now 1. load the preprocessed features, load the reduced feature set, 
        # exclude use of grid serach loso cv, loso cross validation, and leave one subject out
        # and instead use the best hyper param config obtained from summarization.ipynb and train the model
        # not on a specific fold or set of subjects but all subjects
        train_final_estimator(
            subjects_lstm_features,
            subjects_lstm_labels, 
            selector_config=args.pl,
            estimator_name=args.m,
            estimator=models[args.m]['model'],
            hyper_param_config=hyper_param_config
        )