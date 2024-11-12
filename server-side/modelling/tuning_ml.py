#from sklearnex.svm import SVC
#from sklearnex.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools
import json
import numpy as np
import ast
import re
import seaborn as sb

from utilities.loaders import concur_load_data, save_lookup_array, load_lookup_array, save_model

from argparse import ArgumentParser



def sample_ids(subjects_features, n_rows_to_sample):
    """
    args:
        n_rows_to_sample - 
    """

    n_rows_to_sample = n_rows_to_sample if n_rows_to_sample != None else subjects_features.shape[0]
    sampled_ids = np.random.choice(subjects_features.shape[0], size=n_rows_to_sample)

    return sampled_ids

def select_features(subjects_features: pd.DataFrame,
    subjects_labels: pd.DataFrame,
    estimator_name: str,
    selector_config: str,
    n_features_to_select: int,
    class_weights: dict,
    sampled_ids: list | pd.Series | np.ndarray):

    """
    args:
        subjects_features - 
        subjects_labels - 
        selector_config - 
        n_features_to_select - 
        sample_ids - 
    """
    
    # if selected features already have been selected by RFE and saved
    # load it and return it from function
    if load_lookup_array(f'./data/Artifact Detection Data/reduced_{selector_config}_{estimator_name}_feature_set.txt') != False:
        # if features have already been saved load it
        selected_feats = load_lookup_array(f'./data/Artifact Detection Data/reduced_{selector_config}_{estimator_name}_feature_set.txt')

        return selected_feats + ['subject_id']
    
    # remove subject_id column then convert to numpy array
    X = subjects_features.loc[sampled_ids, subjects_features.columns != 'subject_id'].to_numpy()
    Y = subjects_labels.loc[sampled_ids, subjects_labels.columns != 'subject_id'].to_numpy().ravel()

    # select best features first by means of backward
    # feature selection based on support vector classifiers
    model = RandomForestClassifier(class_weight=class_weights, verbose=1)
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, verbose=1)

    # train feature selector on data
    selector.fit(X, Y)

    # obtain feature mask boolean values, and use it as index
    # to select only the columns that have been selected by BFS 
    feats_mask = selector.get_support().tolist()
    subjects_features_cols = subjects_features.columns[subjects_features.columns != 'subject_id']
    selected_feats = subjects_features_cols[feats_mask].to_list()
    print(f"selected features: {selected_feats}")

    # create and save a .txt file containing the selected features by RFE
    save_lookup_array(f'./data/Artifact Detection Data/reduced_{selector_config}_{estimator_name}_feature_set.txt', selected_feats)

    # append also True element to feature mask since subject id
    # has been removed in X
    return selected_feats + ['subject_id']

def leave_one_subject_out(features: pd.DataFrame, labels: pd.DataFrame, subject_id: int, selector_config: str):
    """
    args:
        features - 
        labels - 
        subject_id - id of the subject to leave out from the set 
        of subjects features and set of subjects labels
        selector_config - what authors scaler would be used to scale
        features. Hossain et al. (2022) for instance has used a StandardScaler()
        to normalize the features into smaller values for computational efficiency
    """

    # create boolean values
    cross_set = features['subject_id'] == subject_id
    
    # split train and cross data based on selected subject
    # (14, 1), (14,). At this point features and labels df's 
    # will still contain subject_id column so remove them
    cross_features = features.loc[cross_set, features.columns != 'subject_id'].to_numpy()
    cross_labels = labels.loc[cross_set, labels.columns != 'subject_id'].to_numpy().ravel()
    train_features = features.loc[~cross_set, features.columns != 'subject_id'].to_numpy()
    train_labels = labels.loc[~cross_set, labels.columns != 'subject_id'].to_numpy().ravel()
    

    if selector_config == "hossain":
        # when hossain is picked as the pipeline we need some way to
        # scale/normalize the train features, transform the cross features
        # on the scaler fitted on the train features, this means at every
        # fold or split we must fit a scaler on the train features
        scaler = StandardScaler()

        # scale train and cross features according to train features
        print("scaling features")
        train_features = scaler.fit_transform(train_features)
        cross_features = scaler.transform(cross_features)

        # note: we do not need to save the scaler here as long
        # as the mode is set to "tuning" but if during training
        # the scaler must be saved
    
    return train_features, train_labels, cross_features, cross_labels 

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

def loso_cross_validation(subjects_features: pd.DataFrame,
    subjects_labels: pd.DataFrame,
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
        train_features, train_labels, cross_features, cross_labels = leave_one_subject_out(subjects_features, subjects_labels, subject_id, selector_config)

        # train model
        model.fit(train_features, train_labels)

        # compare true cross and train labels to pred cross and train labels
        pred_train_labels = model.predict(train_features)
        pred_cross_labels = model.predict(cross_features)
        pred_train_probs = model.predict_proba(train_features)
        pred_cross_probs = model.predict_proba(cross_features)

        # compute performance metric values for each fold
        fold_train_acc = accuracy_score(y_true=train_labels, y_pred=pred_train_labels)
        fold_cross_acc = accuracy_score(y_true=cross_labels, y_pred=pred_cross_labels)
        fold_train_prec = precision_score(y_true=train_labels, y_pred=pred_train_labels, average='weighted')
        fold_cross_prec = precision_score(y_true=cross_labels, y_pred=pred_cross_labels, average='weighted')
        fold_train_rec = recall_score(y_true=train_labels, y_pred=pred_train_labels, average='weighted')
        fold_cross_rec = recall_score(y_true=cross_labels, y_pred=pred_cross_labels, average='weighted')
        fold_train_f1 = f1_score(y_true=train_labels, y_pred=pred_train_labels, average='weighted')
        fold_cross_f1 = f1_score(y_true=cross_labels, y_pred=pred_cross_labels, average='weighted')
        # fold_train_roc_auc = roc_auc_score(y_true=train_labels, y_score=pred_train_labels, average='weighted', multi_class='ovo')
        # fold_cross_roc_auc = roc_auc_score(y_true=cross_labels, y_score=pred_cross_labels, average='weighted', multi_class='ovo')
        fold_train_roc_auc_prob = roc_auc_score(y_true=train_labels, y_score=pred_train_probs[:, 1], average='weighted', multi_class='ovo')
        fold_cross_roc_auc_prob = roc_auc_score(y_true=cross_labels, y_score=pred_cross_probs[:, 1], average='weighted', multi_class='ovo')
        
        # save append each metric value to each respective list
        folds_train_acc.append(fold_train_acc)
        folds_cross_acc.append(fold_cross_acc)
        folds_train_prec.append(fold_train_prec)
        folds_cross_prec.append(fold_cross_prec)
        folds_train_rec.append(fold_train_rec)
        folds_cross_rec.append(fold_cross_rec)
        folds_train_f1.append(fold_train_f1)
        folds_cross_f1.append(fold_cross_f1)
        # folds_train_roc_auc.append(fold_train_roc_auc)
        # folds_cross_roc_auc.append(fold_cross_roc_auc)
        folds_train_roc_auc_prob.append(fold_train_roc_auc_prob)
        folds_cross_roc_auc_prob.append(fold_cross_roc_auc_prob)

        print(f"fold: {subject_id} with hyper params: {hyper_param_config} \
              \ntrain acc: {fold_train_acc} cross acc: {fold_cross_acc} \
              \ntrain prec: {fold_train_prec} cross prec: {fold_cross_prec} \
              \ntrain rec: {fold_train_rec} cross rec: {fold_cross_rec} \
              \ntrain f1: {fold_train_f1} cross f1: {fold_cross_f1} \
              \ntrain roc_auc_prob: {fold_train_roc_auc_prob} cross roc_auc_prob: {fold_cross_roc_auc_prob}")
        # \ntrain roc_auc: {fold_train_roc_auc} cross roc_auc: {fold_cross_roc_auc} \

    # once all fold train and cross metric values collected update read
    # dictionary with specific hyper param config as key and its recorded
    # metric values as value
    results[f'{estimator_name}'][hyper_param_config_key] = {
        'folds_train_acc':  folds_train_acc,
        'folds_cross_acc': folds_cross_acc,
        'folds_train_prec': folds_train_prec,
        'folds_cross_prec': folds_cross_prec,
        'folds_train_rec': folds_train_rec,
        'folds_cross_rec': folds_cross_rec,
        'folds_train_f1': folds_train_f1,
        'folds_cross_f1': folds_cross_f1,
        # 'folds_train_roc_auc': folds_train_roc_auc,
        # 'folds_cross_roc_auc': folds_cross_roc_auc,
        'folds_train_roc_auc_prob': folds_train_roc_auc_prob,
        'folds_cross_roc_auc_prob': folds_cross_roc_auc_prob
    }

    with open(f'results/{selector_config}_{estimator_name}_results.json', 'w') as file:
        json.dump(results, file)

def k_fold_cross_validation(subjects_features: pd.DataFrame,
    subjects_labels: pd.DataFrame,
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

    X = subjects_features.loc[:, subjects_features.columns != 'subject_id'].to_numpy()
    Y = subjects_labels.loc[:, subjects_labels.columns != 'subject_id'].to_numpy().ravel()
    
    
    model = estimator(**hyper_param_config, verbose=1)

    # recall that ml classifiers cannot classify multiple classes at once like
    # neural networks so multiple classifiers are trained to classify multiple classes
    # in ovo is computationally expensive for large number of classes, works well for
    # binary classes, less prone with class imbalance. ovr is for multi class rather
    # binary classes. In this case we use ovo in roc auc since we have binary classes.
    # Because we also have an imbalance in classes we must use weighted f1, prec, rec
    # and roc auc as these give importance also to minority classes
    scores = cross_validate(
        model, X, Y, 
        cv=5, 
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovo_weighted'],
        return_train_score=True)

    # once all fold train and cross metric values collected update read
    # dictionary with specific hyper param config as key and its recorded
    # metric values as value
    results[f'{estimator_name}'][hyper_param_config_key] = {
        'folds_train_acc':  scores['train_accuracy'].tolist(),
        'folds_cross_acc': scores['test_accuracy'].tolist(),
        'folds_train_prec': scores['train_precision_weighted'].tolist(),
        'folds_cross_prec': scores['test_precision_weighted'].tolist(),
        'folds_train_rec': scores['train_recall_weighted'].tolist(),
        'folds_cross_rec': scores['test_recall_weighted'].tolist(),
        'folds_train_f1': scores['train_f1_weighted'].tolist(),
        'folds_cross_f1': scores['test_f1_weighted'].tolist(),
        'folds_train_roc_auc': scores['train_roc_auc_ovo_weighted'].tolist(),
        'folds_cross_roc_auc': scores['test_roc_auc_ovo_weighted'].tolist(),
    }

    with open(f'results/{selector_config}_{estimator_name}_results.json', 'w') as file:
        json.dump(results, file)


def grid_search_cv(subjects_features: pd.DataFrame,
    subjects_labels: pd.DataFrame,
    subject_to_id: dict,
    selector_config: str,
    n_features_to_select: int,
    n_rows_to_sample: int | None,
    estimator_name: str,
    estimator,
    cv_type: str,
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

    # sample a small part if not all of the dataset
    sampled_ids = sample_ids(subjects_features, n_rows_to_sample)

    # use returned features from select_features()
    class_weights_param = hyper_params.get('class_weight')
    class_weights = class_weights_param[0] if class_weights_param != None else None
    selected_feats = select_features(subjects_features,
        subjects_labels,
        estimator_name,
        selector_config=selector_config,
        n_features_to_select=n_features_to_select,
        class_weights=class_weights,
        sampled_ids=sampled_ids)
    subjects_features = subjects_features[selected_feats].iloc[sampled_ids]
    subjects_labels = subjects_labels.iloc[sampled_ids]

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
        cross_validator = loso_cross_validation if cv_type == "loso" else k_fold_cross_validation
        cross_validator(
            subjects_features, 
            subjects_labels, 
            subject_to_id, 
            selector_config,
            estimator_name,
            estimator,
            hyper_param_config)



def train_final_estimator(subjects_features: pd.DataFrame,
    subjects_labels: pd.DataFrame,
    selector_config: str,
    estimator_name: str,
    estimator,
    exc_lof: bool, 
    hyper_param_config: dict):

    # remove subject id column, convert to numpy array the dataframes
    # then reduce features based on selected features by RFE

    # if reduced feature set does not yet exist then tuning_ml.py must
    # be run first in tuning mode in order to obtain reduced feature set
    # this only applies for ordinary models and not the hybrid svm which can use
    # both loweer order features and higher order features

    # ml based models do not use exc lof so by default will be false
    # lstm-svm model does use exc lof so if used will be true and will
    # go to else but if lstm-model does not use exc lof then exc lof will
    # be false 
    if exc_lof == False:
        if load_lookup_array(f'./data/Artifact Detection Data/reduced_{selector_config}_{estimator_name}_feature_set.txt') == False:
            return
        
        # if features have already been saved load it
        selected_feats = load_lookup_array(f'./data/Artifact Detection Data/reduced_{selector_config}_{estimator_name}_feature_set.txt')

        # select only features based on selector_config argument
        subjects_features = subjects_features[selected_feats]
        subjects_labels = subjects_labels.drop(columns=['subject_id'])
    else:
        subjects_features = subjects_features.drop(columns=['subject_id'])
        subjects_labels = subjects_labels.drop(columns=['subject_id'])

    # display columns   
    print(subjects_features.columns)
    print(subjects_labels.columns)

    # # # visualizing data before training
    # # sb.pairplot(subjects_features.sample(n=20, axis='columns'))
    # # plt.show()

    # remove subject_id column of both dataframes then
    # convert to numpy arrays
    X = subjects_features.to_numpy()
    Y = subjects_labels.to_numpy().ravel()

    if selector_config == "hossain":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        save_model(scaler, f'./saved/misc/{selector_config}_{estimator_name}_scaler.pkl')

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
    
    for hyper_param in hyper_param_list:
        try:
            # extract hyper param name and strip its last occuring underscore
            # extract hyper param value and convert to 
            # appropriate data type using literal evaluator
            key, value = hyper_param.rsplit('_', 1)
            hyper_param_config[key] = ast.literal_eval(value)
            
        except ValueError:
            hyper_param_config[key] = value

        finally:
            print(type(hyper_param_config[key]))
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
    parser.add_argument("--cv_type", type=str, default="loso", help="what type of cross validator to use whether k-fold or leave one subject out")
    parser.add_argument("--hyper_param_list", type=str, default="C_100", nargs="+", help="list of hyper parameters to be used as configuration during training")
    parser.add_argument("--exc_lof", action="store_true", help="boolean whether to include lower  \
        order features in training hybrid lstm svm. Note that feature selection is turned off when this is True")
    args = parser.parse_args()

    # read and load data
    print(args.exc_lof)
    subjects_features, subjects_labels, subjects_names, subject_to_id = concur_load_data(feat_config=args.pl, exc_lof=args.exc_lof)

    cw_obj = compute_class_weight('balanced', classes=subjects_labels['0'].unique(), y=subjects_labels['0'])
    class_weights = dict(enumerate(cw_obj))
    class_weights = {0: 1, 1: 5}
    # print(class_weights)


    # model hyper params
    models = {
        'lr': {
            # used in Taylor et al. (2015) and Hossain et al. (2022)
            'model': LogisticRegression, 
            'hyper_params': {'C': [0.01, 0.1, 1, 10, 100], 'class_weight': [class_weights]}
        },
        'svm': {
            # used in Taylor et al. (2015) and Hossain et al. (2022)
            'model': SVC, 
            'hyper_params': {
                'kernel': ['rbf', 'poly', 'linear'], 
                'C': [1, 10, 100, 1000], 
                'gamma': [0.001, 0.01, 0.1, 1], 
                'probability': [True],
                'degree': [3, 5, 6],
                # this indicates that there is class imbalance of 80% to 20% of negative to positive classes 
                'class_weight': [class_weights]
            }
            # 'model': LinearSVC, 
            # 'hyper_params': {'C': [1, 10, 100, 1000]}
        },
        'rf': {
            # used in Taylor et al. (2015)
            'model': RandomForestClassifier, 
            'hyper_params': {'n_estimators': [200, 400, 600], 'class_weight': [class_weights]}
        },
        'gbt': {
            # used in Hossain et al. (2022)
            'model': GradientBoostingClassifier, 
            'hyper_params': {'n_estimators': [200, 400, 600], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 10]}
        },
    }

    if args.mode.lower() == "tuning":
        
        # do feature selection, hyperparameter tuning, 
        # loso cross validation across all subjects, and
        # save model & results
        grid_search_cv(
            subjects_features, 
            subjects_labels, 
            subject_to_id, 
            selector_config=args.pl,
            n_features_to_select=args.n_features_to_select, 
            n_rows_to_sample=args.n_rows_to_sample,
            estimator_name=args.m,
            estimator=models[args.m]['model'],
            cv_type=args.cv_type,
            hyper_params=models[args.m]['hyper_params'],
        )

    elif args.mode.lower() == "training":
        # build hyper param config dictionary from input
        hyper_param_config = create_hyper_param_config(hyper_param_list=args.hyper_param_list)
        hyper_param_config['class_weight'] = class_weights
        print(hyper_param_config)
        
        # we can just modify this script such that it doesn't loop through hyper param configs anymore and
        # will just only now 1. load the preprocessed features, load the reduced feature set, 
        # exclude use of grid serach loso cv, loso cross validation, and leave one subject out
        # and instead use the best hyper param config obtained from summarization.ipynb and train the model
        # not on a specific fold or set of subjects but all subjects
        train_final_estimator(
            subjects_features,
            subjects_labels, 
            selector_config=args.pl,
            estimator_name=args.m,
            estimator=models[args.m]['model'],
            exc_lof=args.exc_lof,
            hyper_param_config=hyper_param_config
        )