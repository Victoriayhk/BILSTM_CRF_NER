# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs

import options
from data_utils import read_sentences


NAME_GAZETTEER = '../buff/chinese_name.txt'


def count_char_per_tag(words, tags):
    tag_type = {'O': 0}
    for tag in tags:
        if len(tag) > 2 and tag[2:] not in tag_type:
            tag_type[tag[2:]] = len(tag_type)

    char_count = [dict() for i in range(len(tag_type))]
    for word, tag in zip(words, tags):
        itag = 0 if tag == 'O' else tag_type[tag[2:]]
        for char in word:
            if char not in char_count[itag]:
                char_count[itag][char] = 1
            else:
                char_count[itag][char] += 1

    return char_count


def suffix_analysis(words, tags):
    """2017-05-09: 针对中文的前后缀buff
        + 是否是机构名后缀
        + 是否是地名后缀
    """
    pass


def char_analysis(words, tags, char_count=None):
    """一个词的char_feature一共有num_tags维, 每一维表示其包含字符出现在(训练集)
    对应类别的频率累加和.
    """
    if char_count is None:
        char_count = count_char_per_tag(words, tags)

    num_diff_tags = len(char_count)

    char_count_all = [0 for i in range(num_diff_tags)]
    for idx in range(num_diff_tags):
        char_count_all[idx] = sum(char_count[idx].values())

    for i in range(num_diff_tags):
        for char in char_count[i].keys():
            char_count[i][char] = char_count[i][char] * 1.0 / char_count_all[i]

    char_features = []
    for word, tag in zip(words, tags):
        scores = [0.0 for _ in range(num_diff_tags)]
        for i in range(num_diff_tags):
            for char in word:
                if char in char_count[i]:
                    scores[i] += char_count[i][char]
        char_features.append(scores)

    return char_features


def name_analysis(words):
    """中文人名buff
        + 累计所组成的字在常用名中出现的频率
        + 累计所组成的字在常用名中出现的频率(位置一致)
    """
    name_file = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), NAME_GAZETTEER)

    with codecs.open(name_file, mode='r', encoding='utf-8') as fin:
        names = [x.strip() for x in fin.readlines() if x.strip()]
        print('Name loaded from {0}. {1} names in total.'.format(
            name_file, len(names)))

    mlen = max(len(x) for x in names)
    freq_count = [dict() for _ in range(mlen + 1)]

    for name in names:
        for i, char in enumerate(name):
            if char not in freq_count[i]:
                freq_count[i][char] = 1
            else:
                freq_count[i][char] += 1
            if char not in freq_count[-1]:
                freq_count[-1][char] = 1
            else:
                freq_count[-1][char] += 1

    max_name_length = len(freq_count)
    freq_total = [sum(freq_count[i].values()) for i in range(len(freq_count))]

    name_features = []
    for word in words:
        scores = [0.0 for i in range(max_name_length + 1)]
        for i, char in enumerate(word):
            if (i >= max_name_length):
                break
            if char in freq_count[i]:
                scores[i] = (freq_count[i][char] * 1.0 / freq_total[i])
            if char in freq_count[-1]:
                scores[-1] += freq_count[-1][char] * 1.0 / freq_total[-1]
        name_features.append(scores)

    return name_features


def cap_analysis(words):
    """英文buff: 是否全部大写, 是否全部小写, 是否首字母大写"""
    def cap(s):
        if s.lower() == s:
            return 0 * options.opts.init_scale
        elif s.upper() == s:
            return 1 * options.opts.init_scale
        elif s[0].upper() == s[0]:
            return 2 * options.opts.init_scale
        else:
            return 3 * options.opts.init_scale
    return [[cap(word)] for word in words]


def chinese_buffs(words, tags, caps, char_count):
    char_features = char_analysis(words, tags, char_count)
    name_features = name_analysis(words)
    buffs = [x + y for x, y in zip(char_features, name_features)]
    return buffs


def english_buffs(words, tags, caps, char_count):
    char_features = char_analysis(words, tags, char_count)
    buffs = [x + [y] for x, y in zip(char_features, caps)]
    return buffs


def test():
    data_path = "../data/ace_2004/nwire"
    train_file = os.path.join(data_path, "train.text")

    words, tags, caps = read_sentences(train_file)
    char_count = count_char_per_tag(words, tags)
    buffs = chinese_buffs(words, tags, caps, char_count)
    for i, word in enumerate(words[:200]):
        print(word, tags[i], len(buffs[i]))


if __name__ == '__main__':
    test()
