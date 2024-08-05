import os
import pandas as pd
import itertools
import json
import numpy as np

# import tensorflow as tf
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from argparse import ArgumentParser

from utilities.loaders import concur_load_data, save_lookup_array

def select_features(subjects_features: pd.DataFrame, subjects_labels: pd.DataFrame, selector_config: str, n_features_to_select: int, sample_ids: list | pd.Series | np.ndarray):
    # select best features first by means of backward
    # feature selection based on support vector classifiers
    model = SVC() if selector_config == "taylor" else RandomForestClassifier()
    # selector = SequentialFeatureSelector(svc, n_features_to_select=n_features_to_select, direction='backward', scoring='roc_auc')
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, verbose=1)
    
    # remove subject_id column then convert to numpy array
    X = subjects_features.loc[sample_ids, subjects_features.columns != 'subject_id'].to_numpy()
    Y = subjects_labels.loc[sample_ids, subjects_labels.columns != 'subject_id'].to_numpy().ravel()

    # train feature selector on data
    selector.fit(X, Y)

    # obtain feature mask boolean values, and use it as index
    # to select only the columns that have been selected by BFS 
    # append also True element to feature mask since subject id
    # has been removed in X
    feats_mask = selector.get_support().tolist() + [True]
    selected_feats = subjects_features.columns[feats_mask]

    # create and save a .txt file containing the selected features by RFE
    save_lookup_array(f'./results/reduced_{selector_config}_feature_set.txt')

    return selected_feats

def leave_one_subject_out(features: pd.DataFrame, labels: pd.DataFrame, subject_id: int):
    """
    args:
        features - 
        labels - 
        subjects - 
        subject_id - id of the subject to leave out from the set 
        of subjects features and set of subjects labels
    """

    # create boolean values
    cross_set = features['subject_id'] == subject_id
    
    # split train and cross data based on selected subject
    # (14, 1), (14,)
    cross_features = features.loc[cross_set, :].to_numpy()
    cross_labels = labels.loc[cross_set, :].to_numpy().ravel()
    train_features = features.loc[~cross_set, :].to_numpy()
    train_labels = labels.loc[~cross_set, :].to_numpy().ravel()
    
    return train_features, train_labels, cross_features, cross_labels 

def check_file_key(estimator_name, hyper_param_config_key):
    # if json file exists read and use it
    if os.path.exists(f'./results/{estimator_name}_results.json'):
        # read json file as dictionary
        with open(f'./results/{estimator_name}_results.json') as file:
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

    # if file exists or not return a dictionary but if hyper param 
    # config key already exists return from function
    if check_file_key(estimator_name, hyper_param_config_key) != False:
        results = check_file_key(estimator_name, hyper_param_config_key)
    else:
        return
    
    print(results)
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

    # split features and labels into train and cross by 
    # leaving 1 subject out for cross validatoin and the
    # rest for training, iterated for all subjects
    for subject_id in subject_to_id.values():
        # split data by leaving one subject out for testing
        # and the rest for training
        train_features, train_labels, cross_features, cross_labels = leave_one_subject_out(subjects_features, subjects_labels, subject_id)

        # train model
        model.fit(train_features, train_labels)

        # compare true cross and train labels to pred cross and train labels
        pred_train_labels = model.predict(train_features)
        pred_cross_labels = model.predict(cross_features)

        # compute performance metric values for each fold
        fold_train_acc = accuracy_score(y_true=train_labels, y_pred=pred_train_labels)
        fold_cross_acc = accuracy_score(y_true=cross_labels, y_pred=pred_cross_labels)
        fold_train_prec = precision_score(y_true=train_labels, y_pred=pred_train_labels)
        fold_cross_prec = precision_score(y_true=cross_labels, y_pred=pred_cross_labels)
        fold_train_rec = recall_score(y_true=train_labels, y_pred=pred_train_labels)
        fold_cross_rec = recall_score(y_true=cross_labels, y_pred=pred_cross_labels)
        fold_train_f1 = f1_score(y_true=train_labels, y_pred=pred_train_labels)
        fold_cross_f1 = f1_score(y_true=cross_labels, y_pred=pred_cross_labels)
        fold_train_roc_auc = roc_auc_score(y_true=train_labels, y_score=pred_train_labels)
        fold_cross_roc_auc = roc_auc_score(y_true=cross_labels, y_score=pred_cross_labels)
        
        # save append each metric value to each respective list
        folds_train_acc.append(fold_train_acc)
        folds_cross_acc.append(fold_cross_acc)
        folds_train_prec.append(fold_train_prec)
        folds_cross_prec.append(fold_cross_prec)
        folds_train_rec.append(fold_train_rec)
        folds_cross_rec.append(fold_cross_rec)
        folds_train_f1.append(fold_train_f1)
        folds_cross_f1.append(fold_cross_f1)
        folds_train_roc_auc.append(fold_train_roc_auc)
        folds_cross_roc_auc.append(fold_cross_roc_auc)

        print(f"fold: {subject_id} with hyper params: {hyper_param_config} \
              \ntrain acc: {fold_train_acc} cross acc: {fold_cross_acc} \
              \ntrain prec: {fold_train_prec} cross prec: {fold_cross_prec} \
              \ntrain rec: {fold_train_rec} cross rec: {fold_cross_rec} \
              \ntrain f1: {fold_train_f1} cross f1: {fold_cross_f1} \
              \ntrain roc_auc: {fold_train_roc_auc} cross roc_auc: {fold_cross_roc_auc}")

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
        'folds_train_roc_auc': folds_train_roc_auc,
        'folds_cross_roc_auc': folds_cross_roc_auc
    }

    with open(f'results/{estimator_name}_results.json', 'w') as file:
        json.dump(results, file)

def grid_search_loso_cv(subjects_features: pd.DataFrame, subjects_labels: pd.DataFrame, subject_to_id: dict, selector_config: str, n_features_to_select: int, n_rows_to_sample: int | None, estimator_name: str, estimator, hyper_params: dict, ):
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
    n_rows_to_sample = n_rows_to_sample if n_rows_to_sample != None else subjects_features.shape[0]
    sample_ids = np.random.choice(subjects_features.shape[0], size=n_rows_to_sample)

    # use returned features from select_features()
    # selected_feats = select_features(subjects_features, subjects_labels, selector_config=selector_config, n_features_to_select=n_features_to_select, sample_ids=sample_ids)
    # subjects_features = subjects_features[selected_feats].iloc[sample_ids]
    subjects_features = subjects_features.iloc[sample_ids]
    subjects_labels = subjects_labels.drop(columns=['subject_id']).iloc[sample_ids]

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
            subjects_features, 
            subjects_labels, 
            subject_to_id, 
            estimator_name,
            estimator,
            **hyper_param_config)



if __name__ == "__main__":
    """
    1. There are 43 subjects, 33 of which will be used for training & validation and 10 of which will be used for testing 
    2. All 43 subjects signals will undergo extensive feature extraction prior by running feature_engineering.ipynb
    3. The extracted features from each corresponding subjects recorded signals will now be really split into 33 and 10, the other for trianing and the other for testing
    4. These 33 & 10 subject features will be placed inside a training folder & a testing folder
    5. We then load the newly engineered train data
    6. Data is then used for feature selection by using recursive feature elimination to select best features
    6. A grid search loop is then setup using these best features such that all possible hyper param configurations are used
    7. in each iteration LOSO cross validation will be carried out
        ```
        for hyper param 1 in hyperparams 1
            for hyper param 2 in hyperparams 2
                ...
                for hyper param n in hyperparams n
                    loso_cross_validation(features/X, y/labels, subjects, model)
        ```

    8. loso_cross_validation() will
        - use best features for all 33 subjects
        - run a loop tthat will leave one subject out where for all subjects...
            - these will be our folds
            - for each iteration of this loop will a classifier is trained
            - the classifiers score is then recorded at this fold & move on to next iteration
        - average out all scores collected at each "fold", associate it with the best selected features and the hyper param configuration it used
    9. save the best features found by the RFE algorithm
    """

    # read and parse user arguments
    parser = ArgumentParser()
    parser.add_argument("--n_features_to_select", type=int, default=40, help="number of features to select by RFE")
    parser.add_argument("--n_rows_to_sample", type=int, default=None, help="number of rows to sample during feature selection by RFE")
    parser.add_argument("-m", type=str, default='lr', help="model e.g. lr for logistic regression, rf for random forest, svm for support vector machine, gbt for gradient boosted tree, to train and validate ")
    parser.add_argument("-pl", type=str, default="taylor", 
        help="represents what pipeline which involves what feature set must \
        be kept when data is loaded and what model must the feature selector \
        be based on i.e. SVM or RFC Taylor et al, must be used. (2015) for instance \
        has used most statistical features but variable frequency complex demodulation \
        based features are not used unlike in Hossain et al. (2022) study")
    args = parser.parse_args()

    # read and load data
    subjects_features, subjects_labels, subject_to_id = concur_load_data('./data/Artifact Detection Data/train/', feat_config=args.pl)

    # model hyper params
    models = {
        'lr': {
            'model': LogisticRegression, 
            'hyper_params': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'rf': {
            'model': RandomForestClassifier, 
            'hyper_params': {'n_estimators': [200, 400, 600]}
        },
        'svm': {
            'model': SVC, 
            'hyper_params': {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1]}
        },
        'gbt': {
            'model': GradientBoostingClassifier, 
            # 'hyper_params': {'n_estimators': [200, 400, 600], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 10], 'verbose': [1]}
            'hyper_params': {'n_estimators': [200, 400], 'learning_rate': [0.01], 'max_depth': [3], 'verbose': [1]}
        },
    }

    # do feature selection, hyperparameter tuning, 
    # loso cross validation across all subjects, and
    # save model & results
    grid_search_loso_cv(
        subjects_features, 
        subjects_labels, 
        subject_to_id, 
        selector_config=args.pl,
        n_features_to_select=args.n_features_to_select, 
        n_rows_to_sample=args.n_rows_to_sample,
        estimator_name=args.m,
        estimator=models[args.m]['model'],
        hyper_params=models[args.m]['hyper_params'],
    )