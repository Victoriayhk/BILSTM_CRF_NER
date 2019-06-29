# *-* coding: utf-8 *-*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import codecs
import jieba


def yhk_cut(phrase):
    return list(jieba.cut(phrase))


def cut(content):
    to_cuts = []
    to_tags = []

    i, mlen = 0, len(content)
    while i < mlen:
        cur_text = ""
        while i < mlen and content[i] != '{':
            cur_text += content[i]
            i += 1
        if len(cur_text) > 0:
            to_cuts.append(cur_text)
            to_tags.append('O')

        if i < mlen - 1 and content[i] == '{':
            j = i + 2
            while j < mlen and content[j] != '}':
                j += 1
            cur_text = content[i + 2:j].split(':')
            to_cuts.append(cur_text[1])
            to_tags.append(cur_text[0][:3].upper())
            i = j + 2

    # print("\n".join(to_cuts))

    words, tags = [], []
    for phrase, tag in zip(to_cuts, to_tags):
        cuts = [w for w in yhk_cut(phrase) if not bool(re.search(r'\s', w)) and w != '\n']
        words.extend(cuts)
        if tag == 'O':
            tags.extend(['O' for _ in range(len(cuts))])
        else:
            if len(cuts) == 1:
                tags.append('U-' + tag)
            else:
                tags.append('B-' + tag)
                if len(cuts) > 2:
                    tags.extend(['I-' + tag for _ in range(len(cuts) - 2)])
                tags.append('L-' + tag)

    # print(" ".join(words))

    return words, tags


def test(file):
    names = ['train', 'valid', 'test']
    dsize = [1600, 200, 200]

    words_all = []
    tags_all = []
    with codecs.open(file, 'r', 'utf-8') as fin:
        for c, name in enumerate(names):
            to_write = []
            fout = codecs.open(name + '.text', 'w', 'utf-8')
            for i in range(dsize[c]):
                line = fin.readline()
                words, tags = cut(line)
                words_all.extend(words)
                tags_all.extend(tags)
                to_write.append("\n".join(
                    [word + " " + tag for word, tag in zip(words, tags)]))
            fout.write("\n\n".join(to_write))


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    test("../boson/origin.txt")
