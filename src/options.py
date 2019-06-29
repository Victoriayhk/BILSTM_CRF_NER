# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def init(flags):
    global opts

    if flags is None:
        opts = EnglishOptions()

    if flags.language == "zh":
        opts = ChineseOptions()
    else:
        opts = EnglishOptions()

    if flags.data_path and os.path.exists(flags.data_path):
        opts.data_path = flags.data_path
    if flags.restore and os.path.exists(flags.restore):
        opts.restore = flags.restore
    if flags.len_char:
        opts.len_char = flags.len_char


class EnglishOptions(object):
    def __init__(self):
        # which to save
        self.eval_path = "../eval/"
        self.model_path = "../models/"

        # which data
        self.language = 'en'
        self.data_path = "../data/conll_bilou/"

        self.pre_embed = "../corpus/wiki.en.vector"
        self.dim_embed = 100

        # which data type
        self.use_fp16 = False

        self.zero_digit = True

        # total model params
        self.use_char_cnn = False
        self.use_char_lstm = False
        self.use_char = self.use_char_cnn or self.use_char_lstm

        self.use_handcraft = False
        self.use_lstm = True
        self.use_crf = True

        # char embeddings
        self.len_char = 37  # max length of words in conll datasets
        self.char_vocab_size = None
        self.dim_char = 20

        self.keep_prob = 0.5
        self.num_layers = 1
        self.batch_size = 30
        self.len_seq = 35

        self.lstm_cell = "BasicLSTMCell"

        # 训练设置
        self.max_epoch = 5
        self.max_max_epoch = 2000
        self.init_scale = 0.1
        self.max_grad_norm = 5
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.9

        self.num_tags = None
        self.vocab_size = None
        self.dim_handcraft = 1
        self.restore = False


class ChineseOptions(object):
    def __init__(self):
        # which to save
        self.eval_path = "../eval/"
        self.model_path = "../models/"

        # which data
        self.language = 'zh'
        self.data_path = "../boson_bilou/"

        self.pre_embed = "../corpus/wiki.zh.text.vector"
        self.dim_embed = 100

        # which data type
        self.use_fp16 = False

        self.zero_digit = True

        # total model params
        self.use_char_cnn = False
        self.use_char_lstm = False
        self.use_char = self.use_char_cnn or self.use_char_lstm
        self.use_handcraft = True
        self.use_lstm = True
        self.use_crf = True

        # char embeddings
        self.len_char = 24  # max length of word in boson dataset
        self.char_vocab_size = None
        self.dim_char = 25

        self.keep_prob = 0.5
        self.num_layers = 1
        self.batch_size = 20
        self.len_seq = 35

        self.lstm_cell = "BasicLSTMCell"

        # 训练设置
        self.max_epoch = 5
        self.max_max_epoch = 2000
        self.init_scale = 0.1
        self.max_grad_norm = 5
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.9

        self.num_tags = None
        self.vocab_size = None
        self.dim_handcraft = 1

        self.restore = False
