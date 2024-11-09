import itertools
import json
import os
import pandas as pd
import numpy as np
import ast
import re

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Dice, SquaredHinge, Hinge
from tensorflow.keras.metrics import BinaryCrossentropy as bce_metric, BinaryAccuracy, Precision, Recall, F1Score, AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from utilities.loaders import concur_load_data, save_meta_data, split_data
from models.llanes_jurado import LSTM_CNN
from models.cueva import LSTM_SVM, LSTM_FE

from argparse import ArgumentParser

def leave_one_subject_out(subjects_signals: list[np.ndarray], subjects_labels: list[np.ndarray], subject_id: int):
    """
    args:
        signals - 
        labels - 
        subject_id - id of the subject to leave out from the set 
        of subjects features and set of subjects labels
    """

    # make copy first of list then use it to pop, so as to not
    # mutate original list, pop the element with the index that
    # matches the subject id
    copied_signals = subjects_signals[:]
    copied_labels = subjects_labels[:]
    cross_signals = copied_signals.pop(subject_id)
    cross_labels = copied_labels.pop(subject_id)

    # after popping the copied signals and labels would have now
    # turned into our train data and our popped elements from our copied
    # signals and labels would now be our cross validation data, and because
    # unlike our cross_signals and cross_labels directly being np.ndarrays
    # now after popping, we have to concatenate the list of our copied signals
    # and labels which is our train data across the 0th dimension since the 
    # dimension of each element is (m, window_size, 1)
    train_signals = np.concatenate(copied_signals, axis=0)
    train_labels = np.concatenate(copied_labels, axis=0)

    # # this would be appropriate if there was a larger ram
    # scaler = MinMaxScaler()
    # train_signals = scaler.fit_transform(train_signals)
    # cross_signals = scaler.transform(cross_labels)

    return train_signals, train_labels, cross_signals, cross_labels

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

def loso_cross_validation(subjects_signals: list[np.ndarray],
    subjects_labels: list[np.ndarray],
    subject_to_id: dict,
    selector_config: str,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    estimator_name: str,
    estimator: tf.keras.Model,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    **hyper_param_config: dict):
    """
    args:
        subjects_signals: pd.DataFrame - 
        subjects_labels: pd.DataFrame - 
        subject_to_id: dict - 
        model - 
        hyper_param_config: dict - 
    """

    # create key out of hyper_param_config
    hyper_param_config_key = "|".join([f"{hyper_param}_{value}" for hyper_param, value in hyper_param_config.items()])
    print(hyper_param_config_key)

    # if file exists or not return a dictionary but if hyper param 
    # config key already exists return from function
    if check_file_key(selector_config, estimator_name, hyper_param_config_key) != False:
        results = check_file_key(selector_config, estimator_name, hyper_param_config_key)
    else:
        return
    
    # define early stopping callback to stop if there is no improvement
    # of validation loss for 30 consecutive epochs
    stopper = EarlyStopping(
        monitor='val_auc',
        patience=threshold_epochs)
    callbacks = [stopper]

    # initialize empty lists to collect all metric values per fold
    folds_train_loss = []
    folds_train_acc = []
    folds_train_prec = []
    folds_train_rec = []
    folds_train_f1 = []
    folds_train_roc_auc = []

    folds_cross_loss = []
    folds_cross_acc = []
    folds_cross_prec = []
    folds_cross_rec = []
    folds_cross_f1 = []
    folds_cross_roc_auc = []

    # split signals and labels into train and cross by 
    # leaving 1 subject out for cross validatoin and the
    # rest for training, iterated for all subjects
    for subject_id in subject_to_id.values():
        # split data by leaving one subject out for testing
        # and the rest for training
        train_signals, train_labels, cross_signals, cross_labels = leave_one_subject_out(subjects_signals, subjects_labels, subject_id)
        train_labels[train_labels == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0
        cross_labels[cross_labels == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0

        # create/recreate model with specific hyper param configurations
        # in every hyper param config we must define/redefine an optimizer 
        model = estimator(**hyper_param_config)
        optimizer = opt(learning_rate=alpha)
        compile_args = {"optimizer": optimizer, "loss": loss, "metrics": metrics}# if len(metrics) > 0  else {"optimizer": optimizer, "loss": loss} 
        model.compile(**compile_args)

        # begin training model
        history = model.fit(train_signals, train_labels, 
        epochs=training_epochs,
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_data=(cross_signals, cross_labels),
        verbose=1,)

        # # compare true cross and train labels to pred cross and train labels
        # pred_train = model.predict(train_signals)
        # pred_cross = model.predict(cross_signals)

        # if estimator_name.lower() == "lstm-svm":
        #     """still can't accomodate for sigmoid output layers
        #     because we do this sign and casting only if the dense
        #     layers are unactivated like the svm
        #     """
        #     print("converting output of lstm-svm to binary labels...")
        #     # signed_train_labels = tf.sign(pred_train_labels)

        #     # # using cast turns all negatives to -1, zeros to 0,
        #     # # and positives to 1
        #     # pred_train_labels = tf.cast(signed_train_labels >= 1, "float")
        #     # signed_cross_labels = tf.sign(pred_cross_labels)
        #     # pred_cross_labels = tf.cast(signed_cross_labels >= 1, "float")
        #     pred_train_probs = tf.nn.sigmoid(pred_train)
        #     pred_cross_probs = tf.nn.sigmoid(pred_cross)
        
        # accuracy takes in solid 1s and 0s
        # precision takes in solid 1s and 0s
        # recall takes in solid 1s and 0s

        # binary cross entropy takes in probability outputst
        # binary accuracy takes in probability outputs
        # f1 score takes in probability outputs
        # auc takes in probability outputs

        # now both models are able to output probability values
        # what I want is to also save DSC and SquaredHinge losses
        # in the dictionary of results
        fold_train_loss = history.history['loss'][-1]
        fold_cross_loss = history.history['val_loss'][-1]
        fold_train_acc = history.history['binary_accuracy'][-1]
        fold_cross_acc = history.history['val_binary_accuracy'][-1]
        fold_train_prec = history.history['precision'][-1]
        fold_cross_prec = history.history['val_precision'][-1]
        fold_train_rec = history.history['recall'][-1]
        fold_cross_rec = history.history['val_recall'][-1]
        fold_train_f1 = history.history['f1_score'][-1]
        fold_cross_f1 = history.history['val_f1_score'][-1]
        fold_train_roc_auc = history.history['auc'][-1]
        fold_cross_roc_auc = history.history['val_auc'][-1]
        
        # save append each metric value to each respective list
        folds_train_loss.append(fold_train_loss)
        folds_cross_loss.append(fold_cross_loss)
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
              \ntrain loss: {fold_train_loss} cross loss: {fold_cross_loss} \
              \ntrain acc: {fold_train_acc} cross acc: {fold_cross_acc} \
              \ntrain prec: {fold_train_prec} cross prec: {fold_cross_prec} \
              \ntrain rec: {fold_train_rec} cross rec: {fold_cross_rec} \
              \ntrain f1: {fold_train_f1} cross f1: {fold_cross_f1} \
              \ntrain roc_auc: {fold_train_roc_auc} cross roc_auc: {fold_cross_roc_auc}")

    # once all fold train and cross metric values collected update read
    # dictionary with specific hyper param config as key and its recorded
    # metric values as value
    results[f'{estimator_name}'][hyper_param_config_key] = {
        'folds_train_loss': folds_train_loss,
        'folds_cross_loss': folds_cross_loss,
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

    # if complete cross validation of all subjects is not finished 
    # this will not run
    with open(f'results/{selector_config}_{estimator_name}_results.json', 'w') as file:
        json.dump(results, file)

def grid_search_cv(subjects_signals: list[np.ndarray],
    subjects_labels: list[np.ndarray],
    subject_to_id: dict,
    selector_config: str,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    estimator_name: str,
    estimator: tf.keras.Model,
    hyper_params: dict):

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
            subjects_signals, 
            subjects_labels, 
            subject_to_id, 
            selector_config,
            alpha,
            opt,
            loss,
            metrics,
            estimator_name,
            estimator,
            threshold_epochs, 
            training_epochs, 
            batch_size,
            **hyper_param_config)
        


def train_final_estimator(subjects_signals: list[np.ndarray],
    subjects_labels: list[np.ndarray],
    selector_config: str,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    estimator_name: str,
    estimator,
    hyper_param_config: dict):

    """
    args:
        subjects_signals - 
        subjects_labels - 
        alpha - 
        loss - 
        metrics - 
        threshold_epochs - 
        training_epochs - 
        batch_size - 
        estimator_name - 
        estimator - 
        hyper_param_config - 
    """
    # split signal data into training and cross validation
    # as cross validation will be needed in case of early stopping
    # i.e. if validation metrics do not improve anymore as training
    # progresses 
    train_signals, train_labels, cross_signals, cross_labels = split_data(subjects_signals, subjects_labels, test_ratio=0.1)
    
    print(f'train_signals length: {len(train_signals)}')
    print(f'train_labels length: {len(train_labels)}')
    print(f'cross_signals length: {len(cross_signals)}')
    print(f'cross_labels length: {len(cross_labels)}')

    # concatenate both train and cross validation lists of
    # signal and label numpy arrays into a single 
    # combined training and cross validation dataset
    X_trains = np.concatenate(train_signals, axis=0)
    Y_trains = np.concatenate(train_labels, axis=0)
    X_cross = np.concatenate(cross_signals, axis=0) 
    Y_cross = np.concatenate(cross_labels, axis=0)

    cw_obj = compute_class_weight(class_weight='balanced', classes=np.unique(Y_trains), y=Y_trains.ravel())
    class_weights = dict(enumerate(cw_obj))
    print(class_weights)

    # transform train and cross ys to -1 if lstm-svm is chosen
    Y_trains[Y_trains == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0
    Y_cross[Y_cross == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0
    print(f'train_labels unique: {np.unique(Y_trains)}')
    print(f'cross_labels unique: {np.unique(Y_cross)}')

    # # this would be appropriate if there was a larger ram
    # # min max scale training data and scale cross validation
    # # data based on scaler scaled on training data
    # scaler = MinMaxScaler()
    # X_trains = scaler.fit_transform(X_trains)
    # X_cross = scaler.transform(X_cross)
    # save_model(scaler, f'./saved/misc/{selector_config}_{estimator_name}_scaler.pkl')

    # create model with specific hyper param configurations
    model = estimator(**hyper_param_config)

    # in every hyper param config we must define/redefine an optimizer 
    optimizer = opt(learning_rate=alpha)
    compile_args = {"optimizer": optimizer, "loss": loss, "metrics": metrics}# if len(metrics) > 0  else {"optimizer": optimizer, "loss": loss} 
    model.compile(**compile_args)

    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no improvement
    # of validation loss for 10 consecutive epochs
    path = f"./saved/weights/{selector_config}_{estimator_name}"
    info = "_{epoch:02d}_{val_auc:.4f}.weights.h5"
    weights_path = path + info

    # create callbacks
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_auc',
        verbose=1,
        # save_best_only=True,
        save_weights_only=True,
        mode='max')
    stopper = EarlyStopping(
        monitor='val_auc',
        patience=threshold_epochs)
    
    # append callbacks
    callbacks = [checkpoint, stopper]

    # save hyper params and other attributes of model 
    # for later model loading
    save_meta_data(f'./saved/misc/{selector_config}_{estimator_name}_meta_data.json', hyper_param_config)

    # begin training final model, without validation data
    # as all data combining training and validation will all be used
    # in order to increase model generalization on test data
    history = model.fit(X_trains, Y_trains, 
    epochs=training_epochs,
    batch_size=batch_size, 
    callbacks=callbacks,
    class_weight=class_weights,

    # we set the validation split to all possible signal values not 
    # one subject so model can pull knowledge from more training subjects 
    validation_data=(X_cross, Y_cross),
    verbose=1,)

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
    # python tuning_dl.py -m lstm-svm -pl cueva -lr 1e-3 -bs 1024 --mode tuning
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='lstm-cnn', help="model e.g. lstm-cnn for Llanes-Jurado et al. (2023) LSTM-CNN model, lstm-svm for Cueva et al. (2025), to train and validate ")
    parser.add_argument("-pl", "--pipeline", type=str, default="jurado", 
        help="represents what pipeline which involves what feature set must \
        be kept when data is loaded must be used. Jurado et al. (2023) for instance \
        will use mostly the raw signals as features unlike variable frequency complex \
        demodulation based features which are used in Hossain et al. (2022) study")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="what learning rate value should the optimizer of the model use")
    parser.add_argument("-the", "--threshold_epochs", type=int, default=10, help="represents how many epochs should the early stopping callback stop training the model tolerate metrics that aren't improving")
    parser.add_argument("-tre", "--training_epochs", type=int, default=30, help="represents how many epochs should at the maximum the model train regardless whether its metric values improve or not")
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="batch size during model training")
    parser.add_argument("--mode", type=str, default="tuning", help="tuning mode will not save weights during \
        fitting and while in training mode saves weights")
    parser.add_argument("--hyper_param_list", type=str, default="window_size_640", nargs="+", help="list of hyper parameters to be used as configuration during training")
    args = parser.parse_args()

    # read and load data
    # will be a list of 3D numpy arrays
    subjects_signals, subjects_labels, subjects_names, subject_to_id = concur_load_data(feat_config=args.pipeline)

    # # create and load test data
    # m = 1000
    # window_size = 320
    # n_f = 1
    # n_subjects = 5
    # subjects_signals = [np.random.randn(m, window_size, 1) for _ in range(n_subjects)]
    # subjects_labels = [np.random.randint(low=0, high=2, size=(m, 1)) for _ in range(n_subjects)]
    # subject_to_id = {subject: id for id, subject in enumerate(range(n_subjects))}

    # model hyper params
    models = {
        'lstm-svm': {
            'model': LSTM_SVM, 
            'hyper_params': {
                'window_size': [5 * 128], 
                'n_a': [16, 32], 
                'drop_prob': [0.05, 0.1, 0.75], 
                'C': [0.7, 1, 10], 
                'gamma': [0.01, 0.1, 0.5, 1], 
                'units': [10]
            },
            'opt': Adam,
            'loss': Hinge(),

            # following metrics would not work since all these require z to be activated by the sigmoid activation function
            # and naturally comparing Y_true which are 1's and 0's to unactivated values like 1.23, 0.28, 1.2, etc. will result
            # in 0 metric values being produced from Precision, Recall, etc.
            'metrics': [bce_metric(), BinaryAccuracy(), F1Score(), AUC(name='auc')]
        },
        'lstm-cnn': {
            'model': LSTM_CNN,
            'hyper_params': {
                'window_size': [5 * 128], 
                'n_a': [16, 32],
                'drop_prob': [0.05, 0.75],
                'filter_size': [32],
                'kernel_size': [5]
            },
            'opt': RMSprop,
            'loss': Dice(),
            'metrics': [bce_metric(), BinaryAccuracy(), F1Score(), AUC(name='auc')]
        },
        'lstm-fe': {
            'model': LSTM_FE,
            'hyper_params': {
                'window_size': [5 * 128], 
                'n_a': [16, 32], 
                'drop_prob': [0.05, 0.1, 0.75], 
                'lamb_da': [0.1]
            },
            'opt': Adam, 
            'loss': Dice(),
            'metrics': [bce_metric(), BinaryAccuracy(), F1Score(), AUC(name='auc'), Precision(), Recall()]
        }
    }

    if args.mode.lower() == "tuning":
        # do feature selection, hyperparameter tuning, 
        # loso cross validation across all subjects, and
        # save model & results
        grid_search_cv(
            subjects_signals, 
            subjects_labels, 
            subject_to_id,
            selector_config=args.pipeline,
            alpha=args.learning_rate,
            opt=models[args.model]['opt'],
            loss=models[args.model]['loss'],
            metrics=models[args.model]['metrics'],
            threshold_epochs=args.threshold_epochs,
            training_epochs=args.training_epochs,
            batch_size=args.batch_size,
            estimator_name=args.model,
            estimator=models[args.model]['model'],
            hyper_params=models[args.model]['hyper_params'],
        )

    elif args.mode.lower() == "training":
        # build hyper param config dictionary from input
        hyper_param_config = create_hyper_param_config(hyper_param_list=args.hyper_param_list)
        print(hyper_param_config)
        
        # we can just modify this script such that it doesn't loop through hyper param configs anymore and
        # will just only now 1. load the preprocessed features, load the reduced feature set, 
        # exclude use of grid serach loso cv, loso cross validation, and leave one subject out
        # and instead use the best hyper param config obtained from summarization.ipynb and train the model
        # not on a specific fold or set of subjects but all subjects
        train_final_estimator(
            subjects_signals,
            subjects_labels, 
            selector_config=args.pipeline,
            alpha=args.learning_rate,
            opt=models[args.model]['opt'],
            loss=models[args.model]['loss'],
            metrics=models[args.model]['metrics'],
            threshold_epochs=args.threshold_epochs,
            training_epochs=args.training_epochs,
            batch_size=args.batch_size,
            estimator_name=args.model,
            estimator=models[args.model]['model'],
            hyper_param_config=hyper_param_config
        )