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
import pkg_resources
from symspellpy import SymSpell, Verbosity

try:
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
except:
    nltk.download('all')
    LEMMATIZER = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
SYM_SPELL = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
SYM_SPELL.load_dictionary(
    pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt"),
    term_index=0,
    count_index=1
)

MISSING_IN_VOCAB = '<mvt>'
MISSING_IDX = 0
PADDING = '<pad>'
PADDING_IDX = 1

VOCAB_LOCATION = 'processed/vocab.joblib'
ENCODED_LOCATION = 'processed/encoded.joblib'

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
    result = []
    for word in word_tokenize(text):
        if word in STOP_WORDS:
            continue
        # attempt to fix spelling errors
        suggestions = SYM_SPELL.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True
        )
        result.append(LEMMATIZER.lemmatize(suggestions[0].term))
    
    return result


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
    encoded.insert(0, min(len(text_array), max_length))
    return np.array(encoded).reshape(1, -1)

def decode_text(encoded_text, vocab):
    # horrible and slow. i should have kept a mapping back to the original input!
    text_tok = []
    for i in encoded_text[1:]:
        if i == vocab[PADDING]:
            break
        for k, v in vocab.items():
            if i == v:
                text_tok.append(k)
                break
                
    
    return text_tok


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

