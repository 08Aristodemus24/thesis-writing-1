from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import requests
from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError, NameResolutionError
import json
from datetime import datetime as dt

# ff. imports are for getting secret values from .env file
from pathlib import Path
import os

# import and load model architectures as well as decoder
from modelling.models.cueva import LSTM_SVM
from modelling.models.llanes_jurado import LSTM_CNN
# from modelling.utilities.preprocessors import decode_predictions, map_value_to_index, preprocess
from modelling.utilities.loaders import load_meta_data, load_model

import tensorflow as tf

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000", "https://gen-philo-text.vercel.app", "https://gen-philo-text.onrender.com"])

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



load_miscs()
load_models()

print(models['lstm-cnn']['model'].get_config())
print(models['lstm-svm']['model'].get_config())



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

