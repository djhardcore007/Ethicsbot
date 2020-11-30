#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Usage: python3 multi_clf.py raw_data_dir output_dir
    Only train when you have gpu ready

"""

import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm_notebook
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
from tensorflow.python.client import device_lib
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from simpletransformers.classification import ClassificationModel
import logging
import datetime


def load_data(data_dir):
    raw_data = pd.read_csv(data_dir, usecols=[1, 2, 3, 4, 5, 6])
    return raw_data


def tokenize(sent):
    """
    remove punc and lower case
    """
    token_lst = sent.lower().translate(str.maketrans('', '', string.punctuation)).split()
    return token_lst


def text_cleaning(features):
    """
    simple replacement of abbr and new_line characters
    """
    features = features.replace("\r\n", " ", regex=True)
    features = features.replace("\n", " ", regex=True)
    features = features.replace("\[deleted\]", "", regex=True)
    features = features.replace("\[removed\]", "", regex=True)
    features = features.replace("NTA", "not the asshole", regex=True)
    features = features.replace("YTA", "you are the asshole", regex=True)
    features = features.replace("ESH", "everybody sucks here", regex=True)
    features = features.replace("NAH", "no assholes here", regex=True)
    features = features.replace("OP", "original poster", regex=True)
    features = features.replace("AITA", "Am I the asshole", regex=True)
    features = features.replace("WIBTA", "Would I be the asshole", regex=True)
    features = features.replace("BF", "boyfriend", regex=True)
    features = features.replace("GF", "girlfriend", regex=True)
    return features


def get_text_length(submissions):
    # tokenize
    token_lst = submissions.apply(lambda x: tokenize(x))
    # remove questions that are shorter than 15 words
    sent_len = token_lst.apply(lambda x: len(x))
    print("max/average/min sentence length: {},{},{}".format(np.max(sent_len), np.mean(sent_len), np.min(sent_len)))
    return sent_len


def preprocess(data):
    # drop na values
    D = data[data.label1.isna() == False].dropna()
    print("There are only {} lines left after dropping NA lines".format(D.shape[0]))
    # concat title and text
    submissions = [D.Title.iloc[i] + " " + D.Text.iloc[i] for i in range(D.shape[0])]
    submissions = pd.Series(submissions, name="submission")
    # cleaning submission
    submissions = text_cleaning(submissions)
    # cleaning comments
    comments = text_cleaning(D.Comment)
    # label encoding
    le = preprocessing.LabelEncoder()
    le.fit(["NTA", "YTA", "NAH", "ESH"])
    y = le.transform(D.label1)
    y = pd.Series(y)
    # get sentence length
    sent_len = get_text_length(submissions)
    # concat
    dat_ = pd.DataFrame(pd.concat([submissions, sent_len, comments, y]).values.reshape(4, -1).T,
                        columns=["submission", "submission_length", "comment", "y"])
    # maintain submission sentence length > 15 words
    dat_ = dat_.sort_values(by=["submission_length"])
    cleaned_data = dat_[dat_.submission_length > 15]
    print("Cleaned data shape: {}".format(cleaned_data.shape))
    return cleaned_data


def select_submission_length(data, lower_len, higher_len):
    return data[lower_len < data.sent_length][data.sent_length <= higher_len]


def make_train_test_df(subset):
    X_res, y_res = subset.text.values, subset.label.values
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, shuffle=True)     # train/dev split
    train_df = pd.DataFrame({'text': X_train.reshape(-1), 'label': y_train})
    eval_df = pd.DataFrame({'text': X_test.reshape(-1), 'label': y_test})
    return train_df, eval_df


def train(train_df, max_sub_len, output_dir):
    model_type = 'distilbert'
    lr = 2e-5
    sent_length = max_sub_len
    OUTPUT_DIR = output_dir \
                 + str(datetime.datetime.now())[:19] + '_' + model_type + '_' + str(sent_length) + '_' + str(lr)
    print("model is saved at: {}".format(OUTPUT_DIR))
    training_config = {'output_dir': OUTPUT_DIR,
                       'reprocess_input_data': True,
                       'overwrite_output_dir': True,
                       'num_train_epochs': 2,
                       'train_batch_size': 32,
                       'eval_batch_size': 32,
                       'learning_rate': lr,
                       'max_seq_length': sent_length}
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model = ClassificationModel(model_type, 'distilbert-base-cased', num_labels=4, args=training_config)
    torch.cuda.empty_cache()
    model.train_model(train_df)
    return model


def evaluate(model, eval_df):
    return model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # load data
    data = load_data(data_dir)
    cleaned_data = preprocess(data)
    cleaned_data.y, cleaned_data.submission_length = cleaned_data.y.astype(
        'int'), cleaned_data.submission_length.astype('int')

    # simple random oversampling
    X, y = cleaned_data.submission.values.reshape(-1, 1), cleaned_data.y
    print('Original dataset shape %s' % Counter(y))
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    print(X_res.shape, y_res.shape)

    # divide data into buckets by submission length
    text = pd.DataFrame(np.concatenate((X_res, y_res.reshape(-1, 1)), axis=1), columns=["text", "label"])
    text["sent_length"] = text.text.apply(lambda x: len(x.split()))
    text_128 = select_submission_length(text, 10, 128)
    text_256 = select_submission_length(text, 128, 256)
    text_400 = select_submission_length(text, 256, 400)
    text_512 = select_submission_length(text, 400, 512)

    # train models separately using different subsets
    sub_len_dict = {"128": text_128, "256": text_256, "400": text_400, "512": text_512}
    for max_sub_len in sub_len_dict:
        print("Start training submission length = {}".format(int(max_sub_len)))
        train_df, eval_df = make_train_test_df(sub_len_dict[max_sub_len])
        model = train(train_df, int(max_sub_len), output_dir)
        result, model_outputs, wrong_predictions = evaluate(model, eval_df)
        print("Training for submission len = {} done!".format(int(max_sub_len)))
