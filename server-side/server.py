from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError, NameResolutionError

# ff. imports are for getting secret values from .env file
from pathlib import Path
import re
import tensorflow as tf
import pandas as pd
import numpy as np

# import and load model architectures as well as decoder
from modelling.models.cueva import LSTM_FE
from modelling.models.llanes_jurado import LSTM_CNN
from modelling.utilities.preprocessors import correct_signals
from modelling.utilities.loaders import load_meta_data, load_model, load_lookup_array, charge_raw_data
from modelling.utilities.feature_extractors import extract_features, extract_features_hybrid, extract_features_per_hour

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000", "https://eda-signal-classifier.vercel.app"])

# global variables
models = {
    'cueva_second_phase-1-5-weighted-svm':{

    },
    'cueva_second_phase-1-9-weighted-svm':{

    },
    'cueva_second_phase-1-2-weighted-svm':{

    },
    'cueva-lstm-fe': {
        # 'model':
        # 'hyper_params':
    },
    'jurado-lstm-cnn': {
        # 'model':
        # 'hyper_params':
    },
    'taylor-svm': {
        # 'model':
        # 'selected_feats':
    },
    'taylor-lr': {
        # 'model':
        # 'selected_feats':
    },
    'taylor-rf': {
        # 'model':
        # 'selected_feats':
    },
    'hossain-gbt': {
        # 'model':
        # 'selected_feats':
        # 'scaler':
    },
    'hossain-svm': {
        # 'model':
        # 'selected_feats':
        # 'scaler':
    },
    'hossain-lr': {
        # 'model':
        # 'selected_feats':
        # 'scaler':
    }
}


# functions to load miscellaneous variables, hyperparameters, and the model itself
def load_miscs():
    """
    loads miscellaneous variables to be used by the model
    """

    global models

    print('loading miscellaneous...')
    # this is for loading miscellaneous variables for 
    # deep learning models such as hyper parameters
    lstm_fe_hp = load_meta_data('./modelling/saved/misc/cueva_lstm-fe_meta_data.json')
    lstm_cnn_hp = load_meta_data('./modelling/saved/misc/jurado_lstm-cnn_meta_data.json')

    models['cueva-lstm-fe']['hyper_params'] = lstm_fe_hp
    models['jurado-lstm-cnn']['hyper_params'] = lstm_cnn_hp

    # this is for loading miscellaneous variables for
    # machine learning models such as the reduced feature set
    taylor_lr_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_taylor_lr_feature_set.txt')
    taylor_svm_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_taylor_svm_feature_set.txt')
    taylor_rf_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_taylor_rf_feature_set.txt')
    hossain_lr_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_hossain_lr_feature_set.txt')
    hossain_svm_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_hossain_svm_feature_set.txt')
    hossain_gbt_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_hossain_gbt_feature_set.txt')
    cueva_second_phase_svm_red_feats = load_lookup_array(f'./modelling/data/Artifact Detection Data/reduced_cueva_second_phase_svm_feature_set.txt')

    # pre-load reduced features here so that features don't have to 
    # be loaded every single time user makes a request
    models['taylor-lr']['selected_feats'] = taylor_lr_red_feats
    models['taylor-svm']['selected_feats'] = taylor_svm_red_feats
    models['taylor-rf']['selected_feats'] = taylor_rf_red_feats
    models['hossain-lr']['selected_feats'] = hossain_lr_red_feats
    models['hossain-svm']['selected_feats'] = hossain_svm_red_feats
    models['hossain-gbt']['selected_feats'] = hossain_gbt_red_feats
    models['cueva_second_phase-1-2-weighted-svm']['selected_feats'] = cueva_second_phase_svm_red_feats
    models['cueva_second_phase-1-5-weighted-svm']['selected_feats'] = cueva_second_phase_svm_red_feats
    models['cueva_second_phase-1-9-weighted-svm']['selected_feats'] = cueva_second_phase_svm_red_feats

    print('miscellaneous loaded.')


def load_preprocessors():
    """
    prepares and loads the saved encoders, normalizers of
    the dataset to later transform raw user input from
    client-side
    """
    global models

    print('loading preprocessors...')

    # pre-load here scaler of hossain used during training
    hossain_lr_scaler = load_model('./modelling/saved/misc/hossain_lr_scaler.pkl')
    hossain_svm_scaler = load_model('./modelling/saved/misc/hossain_svm_scaler.pkl')
    hossain_gbt_scaler = load_model('./modelling/saved/misc/hossain_gbt_scaler.pkl')

    models['hossain-lr']['scaler'] = hossain_lr_scaler
    models['hossain-svm']['scaler'] = hossain_svm_scaler
    models['hossain-gbt']['scaler'] = hossain_gbt_scaler

    print('preprocessors loaded.')

def load_models():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """
    global models
    
    print('loading models...')
    # pre load saved weights for deep learning models
    jurado_lstm_cnn = LSTM_CNN(**models['jurado-lstm-cnn']['hyper_params'])
    jurado_lstm_cnn.load_weights('./modelling/saved/weights/EDABE_LSTM_1DCNN_Model.h5')

    # load side task model and convert it to a feature extractor model 
    lstm_fe = LSTM_FE(**models['cueva-lstm-fe']['hyper_params'])
    lstm_fe.load_weights('./modelling/saved/weights/cueva_lstm-fe_21_0.7489.weights.h5')
    lstm_layer_2 = lstm_fe.get_layer('lstm-layer-2')
    lstm_fe_main = tf.keras.Model(inputs=lstm_fe.inputs, outputs=lstm_layer_2.output)

    # # pre load saved machine learning models
    taylor_lr = load_model('./modelling/saved/models/taylor_lr_clf.pkl')
    taylor_svm = load_model('./modelling/saved/models/taylor_svm_clf.pkl')
    taylor_rf = load_model('./modelling/saved/models/taylor_rf_clf.pkl')
    hossain_lr = load_model('./modelling/saved/models/hossain_lr_clf.pkl')
    hossain_svm = load_model('./modelling/saved/models/hossain_svm_clf.pkl')
    hossain_gbt = load_model('./modelling/saved/models/hossain_gbt_clf.pkl')
    cueva_second_phase_1_5_weighted_svm = load_model('./modelling/saved/models/cueva_second_phase_1_5_weighted_svm_clf.pkl')
    cueva_second_phase_1_9_weighted_svm = load_model('./modelling/saved/models/cueva_second_phase_1_9_weighted_svm_clf.pkl')
    cueva_second_phase_1_2_weighted_svm = load_model('./modelling/saved/models/cueva_second_phase_1_2_weighted_svm_clf.pkl')

    # populate dictionary with loaded models
    models['jurado-lstm-cnn']['model'] = jurado_lstm_cnn
    models['cueva-lstm-fe']['model'] = lstm_fe

    models['taylor-lr']['model'] = taylor_lr
    models['taylor-svm']['model'] = taylor_svm
    models['taylor-rf']['model'] = taylor_rf
    models['hossain-lr']['model'] = hossain_lr
    models['hossain-svm']['model'] = hossain_svm
    models['hossain-gbt']['model'] = hossain_gbt
    models['cueva_second_phase-1-5-weighted-svm']['model'] = cueva_second_phase_1_5_weighted_svm
    models['cueva_second_phase-1-9-weighted-svm']['model'] = cueva_second_phase_1_9_weighted_svm
    models['cueva_second_phase-1-2-weighted-svm']['model'] = cueva_second_phase_1_2_weighted_svm

    print('models loaded.')
    



load_miscs()
load_preprocessors()
load_models()

print(models)



# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     raw_data = request.json
#     prompt = [preprocess(raw_data['prompt'])]
#     temperature = float(raw_data['temperature'])
#     T_x = int(raw_data['sequence_length'])
#     print(raw_data)

#     pred_ids = generate(model, prompts=prompt, char_to_idx=char_to_idx, T_x=T_x, temperature=temperature)
#     decoded_ids = decode_predictions(pred_ids, idx_to_char=idx_to_char)

#     return jsonify({'message': decoded_ids})

@app.errorhandler(404)
def page_not_found(error):
    print(error)
    return 'This page does not exist', 404

# upon loading of client side fetch the model names
@app.route('/model-names', methods=['GET'])
def retrieve_model_names():
    """
    flask app will run at http://127.0.0.1:5000 if /
    in url succeeds another string <some string> then
    app will run at http://127.0.0.1:5000/<some string>

    returns json object of all model names loaded upon
    start of server and subsequent request of client to
    this view function
    """

    data = {
        'model_names': list(models.keys())
    }

    # return data at once since no error will most likely
    # occur on mere loading of the model
    return jsonify(data)

@app.route('/send-data', methods=['POST'])
def predict():
    """
    this route will receive clients uploaded .csv file which contains
    the eda signals of a subject 

    this function will then parse and convert the eda signals from dataframe
    to numerical features if ml model is to be used but if dl model then the
    eda signal df is left as it is and passed to trained model

    if dl models are chosen the charge_raw_data() function is used to 
    preprocess the eda signal df and be used as input to the dl model
    and if ml models signals are then used to extract features from
    via extract_features() function

    we will also have to select only the features as selected during tuning

    depending on what model the client chooses the model may be under Hossain
    et al. (2022), Taylor et al. (2015), or Llanes-Jurado et al. (2023) pipeline
    which may or may not have used StandardScaler() during training, if such is
    the case that the user chooses a model under Hossain et al. (2022) then the
    eda signal df which have now been transformed to numerical features must undergo
    further normalization based on the scaler used during training
    """

    # extract raw data from client
    raw_data = request.form
    raw_files = request.files

    model_name = raw_data['model_name']
    spreadsheet_file = raw_files['spreadsheet_file']
    spreadsheet_fname = re.sub(r".csv", "", spreadsheet_file.filename)
    show_raw = raw_data['show_raw']
    show_correct = raw_data['show_correct']
    show_art = raw_data['show_art']

    subject_eda_data = pd.read_csv(spreadsheet_file, sep=';')
    subject_eda_data.columns = ['time', 'raw_signal', 'clean_signal', 'label', 'auto_signal', 'pred_art', 'post_proc_pred_art']
    print(subject_eda_data)
    
    selector_config, estimator_name = model_name.split('-', 1)
    print(selector_config)
    print(estimator_name)

    # this is if deep learning model is chosen
    if selector_config == "hossain" or selector_config == "taylor":
        # load the extracted test features instead as opposed 
        # to using extract-features() method which takes longer to run
        # subject_features, subject_labels = extract_features(subject_eda_data, extractor_fn=extract_features_per_hour)
        subject_features = pd.read_csv(f'./modelling/data/Artifact Detection Data/test/{spreadsheet_fname}_features.csv', index_col=0)
        subject_labels = pd.read_csv(f'./modelling/data/Artifact Detection Data/test/{spreadsheet_fname}_labels.csv', index_col=0)
        print(subject_features.shape)
        print(subject_labels.shape)

        # once features are extracted features selected during
        # tuning will be used in testing as done also during training
        selected_feats = models[model_name]['selected_feats']
        subject_features = subject_features[selected_feats]
        print(subject_features.columns)

        # convert features and labels into numpy matrices
        X = subject_features.to_numpy()
        Y = subject_labels.to_numpy().ravel()

        # if hossain is the researcher chosen the scaler used during training
        # will be used to scale the test subject features
        if selector_config == "hossain":    
            scaler = models[model_name]['scaler']
            X = scaler.transform(X)

        model = models[model_name]['model']
        Y_pred = model.predict(X)
        Y_pred_prob = model.predict_proba(X)
        print(f"predicted Y: {Y_pred}")
        print(f"unique values and counts: {np.unique(Y_pred, return_counts=True)}")
        print(f"true Y: {Y}")
        print(f"unique values and counts: {np.unique(Y, return_counts=True)}")

        # compute performance metric values for test subject
        test_acc = accuracy_score(y_true=Y, y_pred=Y_pred)
        test_prec = precision_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_rec = recall_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_f1 = f1_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_roc_auc = roc_auc_score(y_true=Y, y_score=Y_pred_prob[:, 1], average="weighted", multi_class="ovo")

        print(f"test acc: {test_acc} \
              \ntest prec: {test_prec} \
              \ntest rec: {test_rec} \
              \ntest f1: {test_f1} \
              \ntest roc_auc: {test_roc_auc}")
        

        # next task here is once predictions are out I need tsome way to map the 
        # predictions to correct the artifacts in the raw data

        # take each 0.5s signal and see if that segment is to be corrected or spline or not
        # according to the prediction of the model on the test data

        # so lets say we have our signals we'd have to divide these segments into 

    elif selector_config == "jurado" or (selector_config == "cueva" and estimator_name == "lstm-fe"):
        # pass
        subject_signals, subject_labels = charge_raw_data(subject_eda_data, x_col="raw_signal", y_col='label', scale=True, verbose=True)
        print(f'signals {subject_signals}, shape: {subject_signals.shape}')
        print(f'labels {subject_labels}, shape: {subject_labels.shape}')

        # signals and labels are already numpy matrices
        # assign to X and Y variable for readability
        X = subject_signals
        Y = subject_labels

        # assign model
        model = models[model_name]['model']

        # depending on dl model Y_pred will either be unactivated logits 
        # or sigmoid probabilities
        Y_pred_prob = model.predict(X)

        # when our predictions is 0.2, 0.15, and below which is <= 0.2 then y_pred will be 0
        # when our predictions is 1, 0.5, 0.4, 0.3, 0.21, and above which is > 0.2 then y_pred will be 1
        # why we do this is because of the imbalance of our dataset, and we
        # want to place a threshold of 20% since there our dataset only consists
        # of 20% of positive classes. Note this conversion is to be used in precision and recall metrics
        Y_pred = tf.cast(Y_pred_prob >= 0.2, tf.int64)

        print(f"predicted Y: {Y_pred}")
        print(f"unique values and counts: {np.unique(Y_pred, return_counts=True)}")
        print(f"true Y: {Y}")
        print(f"unique values and counts: {np.unique(Y, return_counts=True)}")

        # compute performance metric values for test subject
        test_acc = accuracy_score(y_true=Y, y_pred=Y_pred)
        test_prec = precision_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_rec = recall_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_f1 = f1_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_roc_auc = roc_auc_score(y_true=Y, y_score=Y_pred_prob, average="weighted", multi_class="ovo")

        print(f"test acc: {test_acc} \
            \ntest prec: {test_prec} \
            \ntest rec: {test_rec} \
            \ntest f1: {test_f1} \
            \ntest roc_auc: {test_roc_auc}")
        
    # this condition is triggered whenever user picks the cueva but uses the lstm-svm as estimator name
    else:
        # extract lower order features of the test data akin to previous ml models
        # subject_lof, subject_labels = extract_features(subject_eda_data, extractor_fn=extract_features_hybrid)
        subject_lof = pd.read_csv(f'./modelling/data/Hybrid Artifact Detection Data/test/{spreadsheet_fname}_lof.csv', index_col=0)
        subject_labels = pd.read_csv(f'./modelling/data/Hybrid Artifact Detection Data/test/{spreadsheet_fname}_labels.csv', index_col=0)
        print(f'lower order features shape: {subject_lof.shape}')
        print(f'labels shape: {subject_labels.shape}')

        # load test signals and use it to extract the higher order features from it
        # subject_signals, subject_labels = charge_raw_data(subject_eda_data, x_col="raw_signal", y_col='label', scale=True, verbose=True)
        # print(f'signals {subject_signals}, shape: {subject_signals.shape}')
        # print(f'labels {subject_labels}, shape: {subject_labels.shape}')

        # # extract higher features
        # lstm_fe_main = models[model_name]['feature_extractor']
        # subject_hof_arr = lstm_fe_main.predict(subject_signals)
        # columns = [f'HOF_{i}' for i in range(1, subject_hof_arr.shape[1] + 1)]
        # subject_hof = pd.DataFrame(subject_hof_arr, columns=columns)
        subject_hof = pd.read_csv(f'./modelling/data/Hybrid Artifact Detection Data/test/{spreadsheet_fname}_hof.csv', index_col=0)
        print(f'higher order features shape: {subject_hof.shape}')

        # concatenate both dataframes of higher and lower features 
        subject_hof_lof = pd.concat([subject_hof, subject_lof], axis=1)
        print(f'full subject_hof_lof shape: {subject_hof_lof.shape}')

        # once features are extracted features selected during
        # tuning will be used in testing as done also during training
        selected_feats = models[model_name]['selected_feats']
        subject_hof_lof = subject_hof_lof[selected_feats]
        print(f'reduced subject_hof_lof shape: {subject_hof_lof.shape}')

        X = subject_hof_lof.to_numpy()
        Y = subject_labels.to_numpy().ravel()

        # assign model
        model = models[model_name]['model']

        # use model for prediction
        Y_pred = model.predict(X)
        Y_pred_prob = model.predict_proba(X)
        print(f"predicted Y: {Y_pred}")
        print(f"unique values and counts: {np.unique(Y_pred, return_counts=True)}")
        print(f"true Y: {Y}")
        print(f"unique values and counts: {np.unique(Y, return_counts=True)}")

        # compute performance metric values for test subject
        test_acc = accuracy_score(y_true=Y, y_pred=Y_pred)
        test_prec = precision_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_rec = recall_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_f1 = f1_score(y_true=Y, y_pred=Y_pred, average="weighted")
        test_roc_auc = roc_auc_score(y_true=Y, y_score=Y_pred_prob[:, 1], average="weighted", multi_class="ovo")

        print(f"test acc: {test_acc} \
              \ntest prec: {test_prec} \
              \ntest rec: {test_rec} \
              \ntest f1: {test_f1} \
              \ntest roc_auc: {test_roc_auc}")


    # once predictions have been extracted from respective models
    # pass to the correct_signals() function
    res_test_df, dict_metrics = correct_signals(Y_pred, subject_eda_data, selector_config, estimator_name)
    print(f'dict metrics: {dict_metrics}')
    print(f'resultant test df: {res_test_df['clean_signal'] == res_test_df['raw_signal']}')
    
    return jsonify({'corrected_df': res_test_df.to_dict("records")})