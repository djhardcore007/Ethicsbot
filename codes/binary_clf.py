#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Usage: python3 binary_clf train_csv_file_path
    Only train when you have gpu ready

"""


import sys
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from simpletransformers.classification import ClassificationModel
import sklearn
import transformers


def tokenize(sent):
    return sent.lower().translate(str.maketrans('', '', string.punctuation)).split()


def clean_data(file_path):
    data = pd.read_csv(file_path, header=0)
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
    features = features.replace("\r\n\r\n", " ", regex=True)
    features = features.replace("AITA", "Am I the asshole", regex=True)
    features = features.replace("WIBTA", "Would I be the asshole", regex=True)
    # label encoding
    D_.label1 = (D_.label1 == "NTA").astype("int")
    y = D_.label1
    # remove questions that are shorter than 15 words
    X = features.apply(lambda x: tokenize(x))
    sent_len = X.apply(lambda x: len(x))
    print("max/average/min sentence length: {},{},{}".format(np.max(sent_len),
                                                             np.mean(sent_len),
                                                             np.min(sent_len)))
    dat = pd.concat([features, pd.Series(y.values), sent_len], axis=1)
    dat_ = pd.DataFrame(dat.values, columns=["x", "y", "len"])
    dat_ = dat_.sort_values(by=["len"])
    cleaned_data = dat_[dat_.len > 15]
    return cleaned_data


def train_test_split(cleaned_data, rand_state, test_percent):
    return train_test_split(cleaned_data.x, cleaned_data.y,
                            test_size=test_percent, random_state=rand_state,shuffle= True)


if __name__ == '__main__':
    file_path = sys.argv[1]

    data = clean_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(data, 42, 0.2)
    train_df = pd.DataFrame({'text': X_train.replace(r'\n', ' ', regex=True), 'label': y_train})
    eval_df = pd.DataFrame({'text': X_test.replace(r'\n', ' ', regex=True), 'label': y_text})
    # Create a TransformerModel
    model = ClassificationModel('bert', 'bert-base-cased-finetuned-mrpc',
                                args={'num_train_epochs': 4,
                                      'overwrite_output_dir': True,
                                      'train_batch_size': 32,
                                      'eval_batch_size': 32,
                                      'learning_rate': 2e-5})
    # Train the model
    model.train_model(train_df)
    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score,
                                                                auc=sklearn.metrics.roc_auc_score)
    print(result)