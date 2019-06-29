# *-* coding: utf-8

import logging
import os.path
import sys
import multiprocessing
import codecs

import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

from id_maps import BasicIdMap


def load_vocab(vocab_file):
    with codecs.open(vocab_file, "r", "utf-8") as f:
        lines = f.readlines()
        words = [line.split()[1] for line in lines]
        dwords = BasicIdMap(words, is_unique=True, default=words[0])
        return dwords


def load_embedding(dwords, vector_file, norm_scale=0.1, name="embed"):
    """导入本地预训练好的word embeddings

    Args:
        dwords: 词到id的双射
        vector_file: 本地word embedding文件
        norm_scale: 归一化数据范围

    Returns:
        embedding: dwords中的一个词一个词向量
    """
    np_file = "".join([vector_file, ".vocab", str(len(dwords)),
                       '.norm', str(norm_scale), '.npy'])
    if os.path.exists(np_file):
        print("Npy embedding found, loading embeddings from {}".format(
            np_file))
        return np.load(np_file)

    # 这一步需要花不少时间，直接读文件会快很多，但有精度损失，待改
    # 直接使用gensim模块来读很慢， 但可以保留较好的精度，不容易出错，
    # 一旦跑过一遍就存起来，下次直接从npy文件中读
    print("Will take some time to load {}".format(vector_file))
    model = KeyedVectors.load_word2vec_format(vector_file, binary=False)
    num_token, dim_embed = len(model.vocab), len(model[model.vocab.keys()[0]])
    print("{} vectors, dimension={}".format(num_token, dim_embed))
    embedding = np.random.random([len(dwords), dim_embed]) * 20 - 10
    find_vector = [False for x in range(len(dwords))]
    min_val, max_val = 10000.0, -10000.0
    for word in dwords.vocab:
        if word in model.vocab:
            idx = dwords.get_id(word)
            embedding[idx] = model[word]
            min_val = min([min_val, np.min(embedding[idx])])
            max_val = max([max_val, np.max(embedding[idx])])
            find_vector[idx] = True

    # normalization
    embedding = norm_scale - (norm_scale * 2) * (
        max_val - embedding) / (max_val - min_val)

    oov_count = sum(find_vector)
    print("{} of {} words has corresponding vector.".format(
        oov_count, len(dwords)))
    print("save embedding vectors to {}".format(np_file))
    np.save(np_file, embedding)

    return embedding


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        data = f.read().split()
    return data


def test(vector_file):
    # dwords = load_vocab(vocab_file)
    # embedding = load_embedding(dwords, vector_file)
    model = gensim.models.KeyedVectors.load_word2vec_format(
        vector_file, binary=False)
    print "vocab_size: ", len(model.vocab)
    maxs, mins = -100, 100
    for x in model.vocab:
        cmax = max(model[x])
        cmin = min(model[x])
        maxs = max([cmax, maxs])
        mins = min([cmin, mins])
    # for x in dwords.vocab[:100]:
    #     if x == '<unk>' or x not in model:
    #         continue
    #     print x, np.array_equal(embedding[dwords.get_id(x)], model[x])
    #     print embedding[dwords.get_id(x)]
    #     print model[x]

    print maxs, mins


def train(corpus_file, vector_file, min_count=5, dim_embed=100):
    print "Run word2vec..."
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    model = Word2Vec(LineSentence(corpus_file), size=dim_embed, window=5,
                     min_count=min_count, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(vector_file, binary=False)


def conll_to_text(conll_file, target_file):
    sentences = []
    sentence = []
    max_sentence = 0
    for line in codecs.open(conll_file, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                max_sentence = max([max_sentence, len(sentence)])
                sentences.append(' '.join(sentence))
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word[0])
    if len(sentence) > 0:
        max_sentence = max([max_sentence, len(sentence)])
        sentences.append(' '.join(sentence))

    with codecs.open(target_file, 'w', 'utf-8') as f:
        f.write('\n'.join(sentences))

    print "maximun sentence: ", max_sentence


if __name__ == '__main__':
    # conll_to_text("conll/train.text", "conll/train.plain.text")
    # train("conll/train.plain.text", "models/conll.train.vector", min_count=1)
    test("models/skipgram.300.txt")
