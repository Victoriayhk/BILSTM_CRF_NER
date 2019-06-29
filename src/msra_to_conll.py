# *-* coding: utf-8 *-*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import jieba
import codecs


def conll_tag(tag):
    return 'O' if tag == 'o' else tag.upper()


def yhk_cut(phrase):
    return list(jieba.cut(phrase))

def tag_array(tag, count=1):
    if tag == 'O':
        return ['O' for _ in range(count)]
    elif count == 1:
        return ['U-' + tag]
    else:
        arr = ['B-' + tag]
        if count > 2:
            arr.extend(['I-' + tag for _ in range(count - 2)])
        arr.append('L' + tag)
        return arr


def re_seg(sentence):
    i, new_sentence = 0, []
    while i < len(sentence):
        j, tag = i, sentence[i][1]
        while j < len(sentence) and sentence[j][1] == tag:
            j += 1
        tokens = yhk_cut(''.join([x[0] for x in sentence[i:j]]))
        tags = tag_array(tag, j - i)
        new_sentence.extend([x + ' ' + y for x, y in zip(tokens, tags)])
        i = j
    return new_sentence


def train_to_conll(ofile, nfile):
    if not os.path.exists(ofile):
        print("file %s not exists." % ofile)
        raise ValueError

    sentences = []
    for line in codecs.open(ofile, mode='r', encoding='utf-8'):
        items = line.split()
        sentence = [[item.split('/')[0], conll_tag(item.split('/')[1])] for item in items]
        sentence = re_seg(sentence)
        sentences.append('\n'.join(sentence))

    with codecs.open(nfile, mode='w', encoding='utf-8') as fout:
        fout.write('\n\n'.join(sentences))
        fout.write('\n')


def test_to_conll(ofile, nfile):
    if not os.path.exists(ofile):
        print("file %s not exists." % ofile)
        raise ValueError

    sentences = []
    for line in codecs.open(ofile, mode='r', encoding='utf-8'):
        items = line.split()
        sentence = []
        for item in items:
            tokens = yhk_cut(item.split('/')[0])
            tags = tag_array(conll_tag(item.split('/')[1]))
            sentence.extend([x + ' ' + y for x, y in zip(tokens, tags)])
        sentences.append('\n'.join(sentence))

    with codecs.open(nfile, mode='w', encoding='utf-8') as fout:
        fout.write('\n\n'.join(sentences))
        fout.write('\n')


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    train_to_conll("../data/sighan_msra/train_origin.txt", "../data/sighan_msra/train.text")
    test_to_conll("../data/sighan_msra/test_origin.txt", "../data/sighan_msra/test.text")
