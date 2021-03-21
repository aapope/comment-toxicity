import argparse
import json
import os
import pickle
import sys
import joblib
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import encode_single_input


def model_fn(model_dir):
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        model_info['embedding_dim'],
        model_info['num_lstm_layers'],
        model_info['hidden_dims'],
        model_info['vocab_size']
    )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'vocab.joblib')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = joblib.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')

    data = encode_single_input(input_data, model.word_dict)
    data = torch.from_numpy(data).to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    y_hat = model(data).detach().cpu().numpy()
    result = np.round(y_hat).astype(np.int)

    return result
