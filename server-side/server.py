from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError, NameResolutionError

# ff. imports are for getting secret values from .env file
from pathlib import Path
from datetime import datetime as dt
import io
import os
import csv
import json
import requests
import tensorflow as tf
import pandas as pd

# import and load model architectures as well as decoder
from modelling.models.cueva import LSTM_SVM
from modelling.models.llanes_jurado import LSTM_CNN
# from modelling.utilities.preprocessors import decode_predictions, map_value_to_index, preprocess
from modelling.utilities.loaders import load_meta_data, load_model

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000", "https://eda-signal-classifier.vercel.app"])

# global variables
models = {
    # 'lstm-svm': {
    #     'model':
    #     'hyper_params':
    # },
    # 'lstm-cnn': {
    #     'model':
    #     'hyper_params':
    # },
    # 'svm': {
    #     'model':
    #     'hyper_params':
    # },
    # 'lr': {
    #     'model':
    #     'hyper_params':
    # },
    # 'rf': {
    #     'model':
    #     'hyper_params':
    # },
    # 'gbt': {
    #     'model':
    #     'hyper_params':
    # }
}


# functions to load miscellaneous variables, hyperparameters, and the model itself


def load_miscs():
    """
    loads miscellaneous variables to be used by the model
    """

    global models
    lstm_svm_hp = load_meta_data('./modelling/saved/misc/lstm_svm_meta_data.json')
    lstm_cnn_hp = load_meta_data('./modelling/saved/misc/lstm_cnn_meta_data.json')

    models['lstm-svm'] = {'hyper_params': lstm_svm_hp}
    models['lstm-cnn'] = {'hyper_params': lstm_cnn_hp}

def load_preprocessors():
    """
    prepares and loads the saved encoders, normalizers of
    the dataset to later transform raw user input from
    client-side
    """
    # load here feature set of taylor and hossain and reduce the data 
    # to be uploaded by the user to only these feature sets features

    

def load_models():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """
    lstm_svm = LSTM_SVM(**models['lstm-svm']['hyper_params'])
    lstm_cnn = LSTM_CNN(**models['lstm-cnn']['hyper_params'])

    # load weights
    lstm_svm.load_weights('./modelling/saved/weights/lstm_cnn_jurado_07_0.5200.weights.h5')
    lstm_cnn.load_weights('./modelling/saved/weights/lstm_svm_87_0.5690.weights.h5')

    # populate dictionary with loaded models
    models['lstm-cnn']['model'] = lstm_cnn
    models['lstm-svm']['model'] = lstm_svm



# load_miscs()
# load_models()

# print(models['lstm-cnn']['model'].get_config())
# print(models['lstm-svm']['model'].get_config())



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

# # upon loading of client side fetch the model names
# @app.route('/model-names', methods=['GET'])
# def retrieve_model_names():
#     """
#     flask app will run at http://127.0.0.1:5000 if /
#     in url succeeds another string <some string> then
#     app will run at http://127.0.0.1:5000/<some string>

#     returns json object of all model names loaded upon
#     start of server and subsequent request of client to
#     this view function
#     """

#     data = {
#         'model_names': list(models.keys())
#     }

#     # return data at once since no error will most likely
#     # occur on mere loading of the model
#     return jsonify(data)

@app.route('/send-data', methods=['POST'])
def test_predict_a():
    # extract raw data from client
    raw_data = request.form
    raw_files = request.files
    print(raw_data)
    print(raw_files)

    model_name = raw_data['model_name']
    spreadsheet = raw_files['spreadsheet']

    eda_signal_df = pd.read_csv(spreadsheet, sep=';')
    print(eda_signal_df)

    # convert eda signal df to numerical features if ml model is to be used
    # but if dl model then leave the signal df as it is and pass it to trained
    # model
    # charge signal first via charge_raw_data() and extract features from signal
    # via extract_features()

    # # preprocessing/encoding image stream into a matrix
    # encoded_img = encode_image(image.stream)
    # rescaled_img = standardize_image(encoded_img)
    # print(rescaled_img.max())
    # print(rescaled_img.shape)

    # # predictor
    
    # # reshape the image since the model takes in an (m, 256, 256, 3)
    # # input, or in this case a single (1, 256, 256, 3) input
    # img_shape = rescaled_img.shape
    # reshaped_img = np.reshape(rescaled_img, newshape=(1, img_shape[0], img_shape[1], img_shape[2]))
    
    # # predictor
    # logits = models[0].predict(reshaped_img)

    # # decoding stage
    # Y_preds = activate_logits(logits)
    # Y_preds = decode_one_hot(Y_preds)
    # final_preds = re_encode_sparse_labels(Y_preds, new_labels=['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'])
    # print(final_preds)
    
    # return jsonify({'prediction': final_preds.tolist()})