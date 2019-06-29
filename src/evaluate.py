# *-* coding: utf-8 *-*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import subprocess
import numpy as np

extend_marks = ['<bos>', '<eos>', '<blind_token>']

a_conll_eval_example = """
processed 51103 tokens with 5868 phrases; found: 5543 phrases; correct: 4424.
accuracy:  93.45%; precision:  79.81%; recall:  75.39%; FB1:  77.54
              LOC: precision:  92.89%; recall:  80.87%; FB1:  86.46  1561
             MISC: precision:  84.97%; recall:  65.77%; FB1:  74.15  692
              ORG: precision:  56.69%; recall:  68.07%; FB1:  61.86  1621
              PER: precision:  87.90%; recall:  80.12%; FB1:  83.83  1669
"""


def conll_eval(x, y, pred, eval_path):
    """官方评估当前结果

    Args:
        x: id化的原始输入序列
        y: 目标输出标签序列
        pred: 模型预测的标签序列
        eval_path: directory of conlleval script

    Returns: 没有返回值，有输出
    """
    if not os.path.isdir(eval_path):
        raise ValueError("Wrong conll eval path. (%s)" % eval_path)
    eval_script = os.path.join(eval_path, "conlleval")
    eval_output = os.path.join(eval_path, "eval.output")
    eval_scores = os.path.join(eval_path, "eval.scores")

    sentences, sentence = [], []
    for (xi, yi, predi) in zip(x, y, pred):
        if xi == '<bos>':
            continue
        elif xi == '<eos>':
            sentences.append('\n'.join(sentence))
            sentence = []
        else:
            sentence.append(' '.join([xi, yi, predi]))

    with codecs.open(eval_output, mode='w', encoding='utf-8') as f:
        f.write('\n\n'.join(sentences))

    proc = subprocess.Popen(["sudo {0} < {1}".format(eval_script, eval_output)],
                            stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out)
    with codecs.open(eval_scores, 'w', encoding='utf-8') as f:
        f.write(out)

    eval_lines = out.split('\n')
    return float(eval_lines[1].strip().split()[-1])


def reshape(xs, batch_size, len_seq):
    x = np.reshape(np.array(xs), [-1, batch_size, len_seq])
    x = np.transpose(x, (1, 0, 2))
    x = np.reshape(x, [-1])
    return x


def test(filename):
    from data_loader import ConllLoader
    data = ConllLoader(filename)

    batch_size = 2
    len_seq = 30

    xs, ys = [], []
    for (x, y) in data.iterator("train", batch_size, len_seq):
        xs.append(x)
        ys.append(y)

    x = reshape(xs, batch_size, len_seq)
    y = reshape(ys, batch_size, len_seq)
    print(x.shape, y.shape)
    conll_eval(x, y, y, data.dwords, data.dtags, "./")


if __name__ == "__main__":
    test("../boson/")
