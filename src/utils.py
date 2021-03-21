import re
import string
import os
from collections import defaultdict
import joblib
import pickle

import pandas as pd
import numpy as np

import torch

import nltk
# download the required resources if necessary
try:
    from nltk.stem import WordNetLemmatizer
except:
    # i'm lazy; this was easier than looking up the minimum packages needed
    nltk.download('all')
    from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
except:
    nltk.download('all')
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
MISSING_IN_VOCAB = '<mvt>'
MISSING_IDX = 0
PADDING = '<pad>'
PADDING_IDX = 1

VOCAB_LOCATION = 'processed/vocab.joblib'
ENCODED_LOCATION = 'processed/encoded.joblib'

from model import LSTMClassifier


#=====================

def _preprocessor(text):
    # remove IP addresses (typically found at the end)
    text = re.sub('[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', '', text)
    # convert whitespace to space
    text = re.sub('\n|\t|\r', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()


def _tokenizer(text):
    return [LEMMATIZER.lemmatize(word) for word in word_tokenize(text) if word not in STOP_WORDS]


def _create_vocab(tokenized_array, vocab_length):
    word_counts = defaultdict(int)
    
    # count words
    for document in tokenized_array:
        for word in document:
            word_counts[word] += 1
            
    # convert to {word: number} vocabulary
    sorted_words = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:vocab_length]
    # add 2 to leave room for the 2 special toks
    vocab = {v: i + 2 for i, v in enumerate(sorted_words)}
    vocab[MISSING_IN_VOCAB] = MISSING_IDX
    vocab[PADDING] = PADDING_IDX
    
    return vocab


def encode_single_input(text, vocab, max_length=500):
    text_array = _tokenizer(_preprocessor(text))
    encoded = [vocab[PADDING] for i in range(max_length)]
    for i, word in enumerate(text_array):
        if i >= max_length:
            break
        encoded[i] = vocab.get(word, vocab[MISSING_IN_VOCAB])
    encoded.insert(0, min(len(text_array)), max_length)
    return np.array(encoded).reshape(1, -1)


def encode_text(df, use_cache=True, vocab_length=10000, max_length=500):
    if use_cache and os.path.exists(VOCAB_LOCATION):
        with open(VOCAB_LOCATION, 'rb') as f:
            vocab = joblib.load(f)
        with open(ENCODED_LOCATION, 'rb') as f:
            encoded_text = joblib.load(f)
    else:
        text_array = list(df['comment_text'])
        
        # clean
        for i in range(len(text_array)):
            text_array[i] = _tokenizer(_preprocessor(text_array[i]))

        # encode using vocab
        encoded_text = []
        vocab = _create_vocab(text_array, vocab_length)
        for document in text_array:
            # create document with padding
            encoded_document = [vocab[PADDING] for i in range(max_length)]
            for i, word in enumerate(document):
                if i >= max_length:
                    break
                encoded_document[i] = vocab.get(word, vocab[MISSING_IN_VOCAB])
            # prepend the length
            encoded_document.insert(0, min(len(document), max_length))
            encoded_text.append(encoded_document)
        encoded_text = np.array(encoded_text)

        # cache
        os.makedirs('processed', exist_ok=True)
        with open(VOCAB_LOCATION, 'wb') as f:
            joblib.dump(vocab, f)
        with open(ENCODED_LOCATION, 'wb') as f:
            joblib.dump(encoded_text, f)
            
    return vocab, encoded_text


#=====================

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
