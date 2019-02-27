import os
import word2vec
import pandas as pd
import wget
import zipfile
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from textsplit.tools import get_penalty, get_segments, P_k
from textsplit.algorithm import split_optimal, split_greedy, get_total

corpus_directory = './text8'
corpus_url = 'http://mattmahoney.net/dc/text8.zip'
transcript_path = './son.txt'
segment_len = 4
segmented_text_path = './%s_%d.txt' % ('son', segment_len)


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


def preprocess_text(path):
    with open(path, 'rt') as f:
        text = f.read().replace('Yahoo!', 'Yahoo')\
            .replace('\n\n', '\n')\
            .replace('\n', ' ¤')
    return text


def get_sentenced_vectors(text, sentence_analyzer, model_df):
    sentenced_text = sentence_analyzer.tokenize(text)
    vecr = CountVectorizer(vocabulary=model_df.index)
    return vecr.transform(sentenced_text).dot(model_df)


def get_optimal_segmentation(sentenced_text, sentence_vectors, penalty):
    optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
    # seg_limit is maximum number of sentences in a segment. optional
    segmented_text = get_segments(sentenced_text, optimal_segmentation)
    print('%d sentences, %d segments, avg %4.2f sentences per segment' % (
        len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))
    return segmented_text

