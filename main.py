import os
import word2vec
import pandas as pd
import numpy as np
import wget
import zipfile
import nltk
from sklearn.feature_extraction.text import CountVectorizer

corpus_directory = './text8'
corpus_url = 'http://mattmahoney.net/dc/text8.zip'
transcript_path = './son.txt'


def download_corpus(dir_name, url):
    # be sure your corpus is cleaned from punctuation and lowercased
    if not os.path.exists(dir_name):
        filename = wget.download(url)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(dir_name)


def train_model(filename, corpus_path):
    print("Training model...")
    word2vec.word2vec(corpus_path, filename, cbow=1, iter_=5, hs=1,
                      threads=4, sample='1e-5', window=15, size=200, binary=1)
    print("Training done.")


def convert_model_to_df(model_path):
    model = word2vec.load(model_path)
    return pd.DataFrame(model.vectors, index=model.vocab)


def get_sentence_tokenizer():
    nltk.download('punkt')
    return nltk.data.load('tokenizers/punkt/english.pickle')


train_model('wrdvecs-text8.bin', corpus_directory)