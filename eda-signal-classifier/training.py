import os
import pandas as pd
import re
import itertools
from concurrent.futures import ThreadPoolExecutor
# import tensorflow as tf
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC

def _combine_data(subjects_data):
    subjects_features, subjects_labels = zip(*subjects_data)

    subjects_features = pd.concat(subjects_features, axis=0, ignore_index=True)
    subjects_labels = pd.concat(subjects_labels, axis=0, ignore_index=True)

    return subjects_features, subjects_labels

def concur_load_data(dir: str, feat_set: str="Taylor"):
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

    # list all .csv features and .csv labels in directory
    subject_names = list(set([re.sub(r"_features.csv|_labels.csv", "", file) for file in os.listdir(dir)]))
    subject_to_id = {subject: id for id, subject in enumerate(subject_names)}

    # read all .csv features and .csv labels
    def helper(subject_name: str):
        # read and assign id to the subject 
        subject_features = pd.read_csv(f'{dir}{subject_name}_features.csv', index_col=0)

        """ this can't be since I cant coerce a column that are float values to strictly int zeros"""
        subject_features = subject_features.apply(lambda value: 0 if isinstance(value, float) and int(value) == 0 else value)
        subject_features['subject_id'] = subject_to_id[subject_name]

        subject_labels = pd.read_csv(f'{dir}{subject_name}_labels.csv', index_col=0)
        subject_labels = subject_labels.astype(int)
        subject_labels['subject_id'] = subject_to_id[subject_name]

        print(subject_features)
        print(subject_labels)

        return (subject_features, subject_labels)

    with ThreadPoolExecutor() as exe:
        # return from this will be a list of all subjects
        # features and labels e.g. [(subject1_features.csv, subject1_labels.csv)]
        subjects_data = list(exe.map(helper, subject_names))
        subjects_features, subjects_labels = _combine_data(subjects_data)

    return subjects_features, subjects_labels, subject_to_id

def leave_one_subject_out(features: pd.DataFrame, labels: pd.DataFrame, subject_id: int):
    """
    args:
        features - 
        labels - 
        subjects - 
        subject_id - id of the subject to leave out from the set 
        of subjects features and set of subjects labels
    """

    cross_set = features['subject_id'] == subject_id
    
    # split train and cross data based on selected subject
    cross_features = features[cross_set]
    cross_labels = labels[cross_set]
    train_features = features[~cross_set]
    train_labels = labels[~cross_set]
    
    return train_features, train_labels, cross_features, cross_labels 

def loso_cross_validation(subjects_features: pd.DataFrame, subjects_labels: pd.DataFrame, subject_to_id: dict, **hyper_param_config: dict):
    
    
    # select best features first by means of backward
    # feature selection based on support vector classifiers
    svc = SVC()
    sfs = SequentialFeatureSelector(svc, n_features_to_select=40, direction='backward', scoring='roc_auc')
    X = subjects_features.loc[:, subjects_features.columns != 'subject_id']
    Y = subjects_labels.loc[:, subjects_labels.columns != 'subject_id']
    # sfs.fit(X, Y)

    # obtain feature mask boolean values, and use it as index
    # to select only the columns that have been selected by BFS 
    feats_mask = sfs.get_support()
    print(feats_mask)
    selected_feats = subjects_features.columns[feats_mask]
    subjects_features = subjects_features[selected_feats]
    subjects_labels = subjects_labels[subjects_labels.columns != 'subject_id']

    # split features and labels into train and cross by 
    # leaving 1 subject out for cross validatoin and the
    # rest for training, iterated for all subjects
    for subject_id in subject_to_id.values():
        train_features, train_labels, cross_features, cross_labels = leave_one_subject_out(subjects_features, subjects_labels, subject_id)
        print(f'train features: {train_features}\n')
        print(f'cross features: {cross_features}\n')
        # print(f'train_features shape: {train_features.shape}')
        # print(f'train_labels shape: {train_labels.shape}')
        # print(f'cross_features shape: {cross_features.shape}')
        # print(f'cross_labels shape: {cross_labels.shape}')

def grid_search_loso_cv(subjects_features, subjects_labels, subject_to_id, hyper_params: dict={'n_estimators': [10, 50, 100], 'max_depth': [3]}):
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
        loso_cross_validation(subjects_features, subjects_labels, subject_to_id, **hyper_param_config)



if __name__ == "__main__":
    """
    1. so I have 43 subjects, 33 of which I use for training & validation and 10 of which will be used for testing 
    2. All 43 subjects signals will undergo extensive feature extraction by running feature_engineering.ipynb
    3. the extracted features from each corresponding subjects recorded signals will be now really split into 33 and 10, the other for trianing and the other for testing
    4. these 33 & 10 subject features will be placed inside a training folder & a testing folder
    5. the first and foremost step is loading only the newly engineered train data
    6. setting up a grid search loop such that all possible hyper param configurationsare used 
    7. in each iteration LOSO cross validation will be carried out
        ```
        for hyper param 1 in hyperparams 1
            for hyper param 2 in hyperparams 2
                ...
                for hyper param n in hyperparams n
                    LOSO_CV(features/X, y/labels, subjects, model)
        ```

    8. LOSO_CV will
        - use backward feature selection first and select best features (but this is too computationally expensive when procedure of leaveing out one subject is not yet done)
        - use best features for all 33 subjects
        - run a loop tthat will leave one subject out for all subjects these will
            - these will be our folds
            - for each iteration of this loop will we train our classifier
            - record this classifiers score at this fold & move on to next iteration
        - average out all scores collected at each "fold", associate it with the best selected features and the hyper param configuration it used
    """
    subjects_features, subjects_labels, subject_to_id = concur_load_data('./data/Artifact Detection Data/train/')

    grid_search_loso_cv(subjects_features, subjects_labels, subject_to_id, hyper_params={'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1], 'L2': [0.7, 0.8, 0.9, 1]})