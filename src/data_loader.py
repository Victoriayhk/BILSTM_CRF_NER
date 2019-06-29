# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import options
from data_utils import read_sentences
from id_maps import BasicIdMap
from handcrafts import count_char_per_tag
from handcrafts import chinese_buffs
from handcrafts import english_buffs


def build_char_dict(words, show_count=False):
    """字符映射"""
    char_set = set()
    char_count = 0
    for word in words:
        if word == '<bos>' or word == '<eos>':
            continue
        # word = unicode(word, 'utf-8')
        for char in word:
            char_set.add(char)
        char_count += len(word)

    if show_count:  # 打印统计信息
        print("%d(%d unique) chars in total." % (char_count, len(char_set)))

    # 考虑到中文字很多, 对于字也可能存在unk的情况，加上unk
    return BasicIdMap(list(char_set), is_unique=True, default='<unk>')


def pad_word_chars(words, dchars):
    max_length = options.opts.len_char
    chars = []
    chars_len = []
    for word in words:
        padding = [dchars.blind_id] * (max_length - len(word))
        chars.append([dchars.get_id(c) for c in word] + padding)
        chars_len.append(len(word))
    return chars, chars_len


def words_to_ids(words, tags, dwords, dtags, dchars):
    data = dict()
    data["words"] = [dwords.get_id(word) for word in words]
    data["tags"] = [dtags.get_id(tag) for tag in tags]
    if options.opts.use_char:
        data["chars"], _ = pad_word_chars(words, dchars)

    return data


def file_to_ids(filename, dwords, dtags, dchars):
    words, tags = read_sentences(filename)
    return words_to_ids(words, tags, dwords=dwords, dtags=dtags, dchars=dchars)


class ConllLoader(object):
    """CoNLL NER任务的数据导入"""
    def __init__(self):
        opts = options.opts
        # 测试数据文件地址
        train_file = os.path.join(opts.data_path, "train.text")
        valid_file = os.path.join(opts.data_path, "valid.text")
        test_file = os.path.join(opts.data_path, "test.text")
        if not all((os.path.exists(train_file),
                    os.path.exists(valid_file),
                    os.path.exists(test_file))):
            raise ValueError("Wrong data path.")

        # vocabulary
        words, tags, caps = read_sentences(train_file)
        self.dwords = BasicIdMap(words, is_unique=False, default='<unk>')
        self.dtags = BasicIdMap(tags, is_unique=False, default='O')
        self.dchars = build_char_dict(words) if opts.use_char else None

        # handcraft setting
        if opts.use_handcraft:
            feature_fun = chinese_buffs if opts.language == 'zh' else english_buffs
            char_count = count_char_per_tag(words, tags)

        # creating data
        self.data = dict()
        self.data['train'] = words_to_ids(words, tags, self.dwords, self.dtags, self.dchars)
        if opts.use_handcraft:
            self.data['train']['features'] = feature_fun(words, tags, caps, char_count)

        words, tags, caps = read_sentences(valid_file)
        self.dwords.extend_tokens(words)
        self.data['valid'] = words_to_ids(words, tags, self.dwords, self.dtags, self.dchars)
        if opts.use_handcraft:
            self.data['valid']['features'] = feature_fun(words, tags, caps, char_count)

        words, tags, caps = read_sentences(test_file)
        self.dwords.extend_tokens(words)
        self.data['test'] = words_to_ids(words, tags, self.dwords, self.dtags, self.dchars)
        if opts.use_handcraft:
            self.data['test']['features'] = feature_fun(words, tags, caps, char_count)

        # pring infos
        if self.dwords:
            print("%d unique words (+ <bos>, <eos>, <unk>)" % (len(self.dwords) - 3))
        if self.dtags:
            print("%d unique tags (+ U-BOS, U-EOS) = %s" % (len(self.dtags) - 2, self.dtags.vocab))
        if self.dchars:
            print("%d unique chars (+ <unk>)" % (len(self.dchars) - 1))

    def data_size(self, name):
        return len(self.data[name]['words'])

    @property
    def vocab_size(self):
        return len(self.dwords)

    @property
    def char_vocab_size(self):
        if options.opts.use_char:
            return len(self.dchars)
        else:
            return None

    @property
    def num_tags(self):
        return len(self.dtags)

    @property
    def dim_handcraft(self):
        if options.opts.use_handcraft:
            return len(self.data['test']['features'][0])
        else:
            return None

    def iterator(self, name, batch_size, len_seq):
        """iterator for epoch."""

        if name not in ['train', 'valid', 'test']:
            raise ValueError("Wrong dataset name.")
        data = self.data[name]

        data_len = self.data_size(name)
        batch_len = data_len // batch_size
        words = np.zeros([batch_size, batch_len], dtype=np.int32)
        tags = np.zeros([batch_size, batch_len], dtype=np.int32)

        if options.opts.use_char:
            len_char = len(data['chars'][0])
            chars = np.zeros([batch_size, batch_len, len_char], dtype=np.int32)
        if options.opts.use_handcraft:
            dim_feature = len(data['features'][0])
            features = np.zeros([batch_size, batch_len, dim_feature],
                                dtype=np.int32)

        for i in range(batch_size):
            words[i] = data['words'][batch_len * i: batch_len * (i + 1)]
            tags[i] = data['tags'][batch_len * i: batch_len * (i + 1)]
            if options.opts.use_char:
                chars[i] = data['chars'][batch_len * i: batch_len * (i + 1)]
            if options.opts.use_handcraft:
                features[i] = data['features'][batch_len * i: batch_len * (i + 1)]

        epoch_size = batch_len // len_seq

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or len_seq")

        for i in range(epoch_size):
            x = words[:, i * len_seq: (i + 1) * len_seq]
            y = tags[:, i * len_seq: (i + 1) * len_seq]
            if options.opts.use_char:
                c = chars[:, i * len_seq: (i + 1) * len_seq]
            else:
                c = 0
            if options.opts.use_handcraft:
                f = features[:, i * len_seq: (i + 1) * len_seq]
            else:
                f = 0
            yield (x, y, c, f)

    def next_sentence(self, name):
        """逐句迭代器"""
        if name not in ['train', 'valid', 'test']:
            raise ValueError("Wrong dataset name.")
        data = self.data[name]

        data_len = self.data_size(name)
        print("data_len=", data_len)
        x, y = [], []
        for i in range(data_len):
            word, tag = data['words'][i], data['tags'][i]
            x.append(word)
            y.append(tag)
            if word == self.dwords.get_id('<eos>'):
                nx = np.reshape(np.array(x), [1, len(x)])
                ny = np.reshape(np.array(y), [1, len(y)])
                x, y = [], []
                yield (nx, ny)


if __name__ == "__main__":
    data_path = "../conll_bilou"
    train_file = os.path.join(data_path, "train.text")

    options.init(None)
    options.opts = options.ChineseOptions()

    # 测试逐句迭代器
    data = ConllLoader()
