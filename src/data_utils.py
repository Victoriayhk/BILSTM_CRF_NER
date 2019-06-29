# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import codecs


def zero_digits(s):
    """Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def read_sentences(filename, zero=True):
    """read words and tags from a conll formatted file

    Args:
        filename: spicify the conll formatted file
        zero: whether replace digits with zero
        bos_eos: whether add extra <bos>/<eos> token for sentences

    Returns:
        words: a list of tokens (in the original order) in the first column
        tags: a list of corresponding tags in the last column
    """
    words, tags, caps = [], [], []
    sentence_w, sentence_t, sentence_c = [], [], []

    def add_sentence(sentence_w, sentence_t, sentence_c):
        sentence_w.insert(0, '<bos>')
        sentence_t.insert(0, 'U-BOS')
        sentence_c.insert(0, 3)
        sentence_w.append('<eos>')
        sentence_t.append('U-EOS')
        sentence_c.append(3)
        words.extend(sentence_w)
        tags.extend(sentence_t)
        caps.extend(sentence_c)

    for line in codecs.open(filename, 'r', 'utf8'):
        items = line.split()
        if len(items) >= 2:
            if zero:
                sentence_w.append(zero_digits(items[0].lower()))
            else:
                sentence_w.append(items[0].lower())
            sentence_c.append(cap_feature(items[0]))
            sentence_t.append(items[-1])
        else:
            add_sentence(sentence_w, sentence_t, sentence_c)
            sentence_w, sentence_t, sentence_c = [], [], []
    if len(sentence_w) > 0:
        add_sentence(sentence_w, sentence_t, sentence_c)

    return words, tags, caps


def load_csv(file, column=[0]):
    data = [[] for _ in len(column)]
    for line in codecs.open(file, 'r', 'utf-8'):
        items = line.split(',')
        for i, c in enumerate(column):
            data[i].append(items[c])
    return data
