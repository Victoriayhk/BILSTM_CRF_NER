# *-*coding: utf-8 *-*

import os
import sys
import codecs
import argparse

"""修改NER数据的tag标记方式

目标标记方式有三种: io, bio, bilou

用法:
python tag_scheme.py path_to_datafile --translate2 bilou
"""

# 修改系统编码方式为utf-8(默认为ascii)
reload(sys)
sys.setdefaultencoding('utf-8')


# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument(
    '-t',
    '--translate2',
    type=str,
    default='bio',
    help=('转码模式: io, bio, bilou\n'))
FLAGS, other_argvs = parser.parse_known_args()
input_file = other_argvs[0]
# input_file = 'conll/train.text'
assert os.path.isfile(input_file)


def load_sentences(path):
    """导入conll全部内容, 忽略DOCSTART标记
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2   # 至少两列
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# 导入数据
sentences = load_sentences(input_file)


def translate_to_io(lines, fout):
    """lines specify the phase within a ner tag"""
    tag = lines[0][-1][2:]
    for line in lines:
        fout.write(' '.join(line[:-1]))
        fout.write(' I-' + tag + '\n')


def translate_to_bio(lines, fout):
    """lines specify the phase within a ner tag"""
    tag = lines[0][-1][2:]
    fout.write(' '.join(lines[0][:-1]))
    fout.write(' B-' + tag + '\n')
    for line in lines[1:]:
        fout.write(' '.join(line[:-1]))
        fout.write(' I-' + tag + '\n')


def translate_to_bilou(lines, fout):
    """lines specify the phase within a ner tag, similar to
    translate_to_bio"""
    tag = lines[0][-1][2:]
    fout.write(' '.join(lines[0][:-1]))
    if len(lines) == 1:
        fout.write(' U-' + tag + '\n')
    else:
        fout.write(' B-' + tag + '\n')
        for line in lines[1:-1]:
            fout.write(' '.join(line[:-1]))
            fout.write(' I-' + tag + '\n')
        fout.write(' '.join(lines[-1][:-1]))
        fout.write(' L-' + tag + '\n')


# 指定编码格式
if FLAGS.translate2 == 'io':
    fo = input_file + '.io'
    fout = open(fo, 'w')
    translate_fun = translate_to_io
elif FLAGS.translate2 == 'bio':
    fo = input_file + '.bio'
    fout = open(input_file + '.bio', 'w')
    translate_fun = translate_to_bio
else:
    fo = input_file + '.bilou'
    fout = open(input_file + '.bilou', 'w')
    translate_fun = translate_to_bilou


# 转换编码方式
for sentence in sentences:
    last_start, last_tag = 0, 'O'  # 上一次相同标签开始的下标和标签名
    for i, line in enumerate(sentence):
        tag = line[-1] if line[-1] == 'O' else line[-1][2:]  # 当前行的标签
        if line[-1][0] == 'B' or tag != last_tag:  # 当前行标签与上一行不同
            if last_tag != 'O':  # 对上一个标签进行转码
                translate_fun(sentence[last_start:i], fout)
            last_start = i  # 标记当前标签开始位置
        if tag == 'O':
            fout.write(' '.join(line))
            fout.write('\n')
        last_tag = tag
    fout.write('\n')
fout.close()


print "对NER数据文件 %s tag转码为'%s'完成." % (input_file, FLAGS.translate2)
