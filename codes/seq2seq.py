#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Usage: python3 seq2seq.py data_dir submission_lower_len submission_higher_len comment_max_len output_dir
    Only train when you have gpu ready

"""


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import datetime
from simpletransformers.seq2seq import Seq2SeqModel
import logging


def load_data(data_directory):
    cleaned_data = pd.read_csv(data_directory, usecols=[1, 2, 3, 4])
    # count comment length
    cleaned_data["comment_length"] = cleaned_data.comment.apply(lambda x: len(str(x).split()))
    # remove super long texts
    cleaned_data = cleaned_data[cleaned_data.submission_length<512][cleaned_data.comment_length<512][cleaned_data.comment_length>10]
    return cleaned_data


def select_submission_length(data, lower_len, higher_len):
    return data[lower_len<data.submission_length][data.submission_length<=higher_len]


def print_length(selected_data):
    print("num of samples:", selected_data.shape[0])
    print("Avg:")
    print(selected_data[["submission_length","comment_length"]].mean())


def make_train_eval_df(selected_data, TEST_SIZE=0.2):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(selected_data.submission.values,
                                                        selected_data.comment.values,
                                                        test_size=TEST_SIZE, random_state=42, shuffle= True)
    # make the df fit for simple transformer
    train_df = pd.DataFrame({'input_text': X_train.reshape(-1), 'target_text': y_train})
    eval_df = pd.DataFrame({'input_text': X_test.reshape(-1), 'target_text': y_test})
    return train_df, eval_df


if __name__ == '__main__':
    data_dir = sys.argv[1]
    lower_len, higher_len = sys.argv[2], sys.argv[3]
    max_length_ = sys.argv[4]
    output_dir = sys.argv[5]

    data = load_data(data_dir)
    selected_data = select_submission_length(data, lower_len, higher_len)
    print_length(selected_data)
    train_df, eval_df = make_train_eval_df(selected_data)   # train/val split

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # set params
    model_type = "bart"
    max_seq_length = higher_len     # 128/256/512
    max_length = max_length_        # max sentence length to be generated
    bs = 8
    epochs = 2
    OUTPUT_DIR = output_dir + str(datetime.datetime.now())[:19]+'_'+model_type+'_'+str(max_seq_length)+'_'+str(max_length)
    print("find the model and output here: {}".format(OUTPUT_DIR))
    model_args = {
        "output_dir": OUTPUT_DIR,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": max_seq_length,
        "train_batch_size": bs,
        "num_train_epochs": epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": max_length,
        "manual_seed": 4,
    }

    torch.cuda.empty_cache()    # clear cache before training every time
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,)       # load pre-trained model

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    results = model.eval_model(eval_df)

    # Use the model for prediction
    input_seqs = eval_df.input_text.to_list()
    pred = model.predict(input_seqs)
    pred_df = pd.DataFrame(pred, columns=["prediction"])
    pd.concat([eval_df, pred_df], axis=1).to_csv(OUTPUT_DIR + "/prediction_on_test.csv")

    # print some sample predictions
    sample_input = eval_df.input_text.sample(10).to_list()
    sample_pred = model.predict(sample_input)
    for en, de in zip(sample_input, sample_pred):
        print("-------------")
        print(en)
        print(de)