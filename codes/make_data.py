#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    $ python3 make_data.py aita_training.csv
"""

import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


def tokenize(sent):
    """
    to lowercase and remove punctuation
    # 1. Do we need to do stemming or lemmatizing ?
    # ps = nltk.PorterStemmer()
    # def stem(token_list):
    #     return [ps.stem(word) for word in token_list]
    # 2. Some abbr needs to be transformed such as AITA, gf, bf ...
    """
    return sent.lower().translate(str.maketrans('', '', string.punctuation)).split()


def load_data(path):
    data = pd.read_csv(path, header=0)
    # drop na labels
    data_ = data[data.label1.isna() == False]
    print("There are only {} lines left after dropping na labels".format(data_.shape[0]))
    # examine data skewness
    print("Original labels are {}".format(np.unique(data_.label1.values)))
    print("There are {} YTA labels, and {} NTA labels".format(len(data_[data_.label1 == "YTA"]),
                                                              len(data_[data_.label1 == "NTA"])))
    D = data_[(data_.label1 == "YTA") | (data_.label1 == "NTA")]
    # select useful cols
    D_ = pd.concat([D.Title, D.Text, D.label1], axis=1).dropna()
    # concat title and text
    features = [D_.Title.iloc[i] + " " + D_.Text.iloc[i] for i in range(D_.shape[0])]
    features = pd.Series(features, name="text")
    # tokenize
    X = features.apply(lambda x: tokenize(x))
    # label encoding
    D_.label1 = (D_.label1 == "NTA").astype("int")
    y = D_.label1
    # remove questions that are shorter than 10 words
    sent_len = X.apply(lambda x: len(x))
    print("max/average/min sentence length: {},{},{}".format(np.max(sent_len), np.mean(sent_len), np.min(sent_len)))
    dat = pd.concat([X, pd.Series(y.values), sent_len], axis=1)
    dat_ = pd.DataFrame(dat.values, columns=["x", "y", "len"])
    dat_ = dat_.sort_values(by=["len"])
    cleaned_data = dat_[dat_.len > 10]
    print("Cleaned data shape: {}".format(cleaned_data.shape))
    return cleaned_data


class Vectorizer():
    def __init__(self, max_features):
        self.max_features = max_features
        self.vocab_list = None
        self.token_to_index = None

    def fit(self, data):
        # Create a vocab list, self.vocab_list, using the most frequent "max_features" tokens
        # Create a token indexer, self.token_to_index, that will return index of the token in self.vocab
        all_tokens = [word for token_lst in data for word in token_lst]
        word_count_lst = Counter(all_tokens).most_common(self.max_features)
        self.vocab_list = [word_count_pair[0] for word_count_pair in word_count_lst]
        self.token_to_index = {token: i for i, token in enumerate(self.vocab_list)}

    def transform(self, data):
        # This function transforms text dataset into a matrix, data_matrix
        vectors = []
        for token_lst in tqdm(data):
            vector = []
            for token in self.vocab_list:
                if token in token_lst:
                    vector.append(1)
                else:
                    vector.append(0)
            vectors.append(vector)
        data_matrix = np.asarray(vectors)
        return data_matrix


if __name__ == '__main__':
    file_path = sys.argv[1]

    cleaned_data = load_data(file_path)
    # let's look at the data of the plot
    plt.bar(range(len(cleaned_data)), cleaned_data.len)
    plt.show()
    # build dictionary and vectorize data
    max_features = 2000
    vectorizer = Vectorizer(max_features=max_features)
    # you can choose whether to include test data in here.
    vectorizer.fit(cleaned_data.x)
    texts = vectorizer.transform(cleaned_data.x)
    print(texts.shape)
    vocab = vectorizer.vocab_list
    print(len(vocab))