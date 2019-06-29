# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import options
import gensim_emb
import record
from nn import get_LSTM_cell
from nn import bi_LSTM_layer
from nn import hidden_layer
from nn import CRF_layer
from nn import CRF_loss
from nn import trans_scores
from nn import seq_loss
from nn import CRF_decode_tf
from nn import CNN_layer


class NERModel(object):
    """The NER model"""
    def __init__(self, dwords, is_training, dtype=tf.float32):
        opts = options.opts
        self.batch_size = batch_size = opts.batch_size
        self.len_seq = len_seq = opts.len_seq
        num_tags = opts.num_tags
        self.is_training = is_training
        vocab_size = opts.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, len_seq])
        self._targets = tf.placeholder(tf.int32, [batch_size, len_seq])

        # 词向量: 导入预训练好的 或者 随机初始化
        with tf.device("/cpu:0"):
            if opts.pre_embed and is_training:  # 预训练好的词向量
                record.logging("- use pretrained embedddings from {}".format(
                    opts.pre_embed))
                embedding_np = gensim_emb.load_embedding(   # 导入词向量
                    dwords, opts.pre_embed, norm_scale=opts.init_scale)
                embedding_init = tf.constant_initializer(embedding_np)
                embedding_table = tf.get_variable(
                    "embedding_table", shape=embedding_np.shape,
                    initializer=embedding_init)
                opts.dim_embed = embedding_np.shape[1]
            else:                               # 随机初始化的词向量
                if is_training:
                    record.logging("- use randomized embeddings")
                embedding_table = tf.get_variable(
                    "embedding_table", [vocab_size, opts.dim_embed],
                    dtype=dtype)
            scores = tf.nn.embedding_lookup(embedding_table, self._input_data)

        # dropout after word embedding
        if is_training and opts.keep_prob < 1:
            record.logging("- use dropout for word embedding")
            scores = tf.nn.dropout(scores, opts.keep_prob)

        # 双向LSTM encoding
        if opts.use_lstm:
            if is_training:
                record.logging("- use bidirectional LSTM(%s) layer for encoding" % opts.lstm_cell)
            lstm_keep_prob = opts.keep_prob if is_training else 1.0
            lstm_cell = get_LSTM_cell(int(scores.get_shape()[2]),
                                      lstm_keep_prob, cell_type=opts.lstm_cell)
            f_state, b_state, scores = bi_LSTM_layer(
                scores, lstm_cell, opts.num_layers, dtype=dtype)
            self._initial_forward_state = f_state
            self._initial_backward_state = b_state
        else:
            scores = hidden_layer(scores, num_tags, keep_prob=opts.keep_prob)
            self._initial_forward_state = tf.Variable(0, trainable=False)
            self._initial_backward_state = tf.Variable(0, trainable=False)






        # 使用字符模型(cnn 或者 lstm)
        if opts.use_char:
            self._input_chars = tf.placeholder(
                tf.int32, [batch_size, len_seq, opts.len_char])
            char_embedding_table = tf.get_variable(
                "char_embedding_table",
                [opts.char_vocab_size, opts.dim_char], dtype=dtype)
            chars = tf.nn.embedding_lookup(char_embedding_table, self._input_chars)
        else:
            self._input_chars = tf.Variable(0, trainable=False)

            # CNN字符模型
        if opts.use_char_cnn:
            if is_training:
                record.logging("- use char cnn embedding")
            chars = tf.reshape(chars, [-1, opts.len_char, opts.dim_char, 1])
            chars_embeddings = tf.reshape(CNN_layer(chars, opts.dim_char),
                                          [batch_size, len_seq, opts.dim_char])
            scores = tf.concat(2, [scores, chars_embeddings])

        # LSTM字符模型
        if opts.use_char_lstm:
            if is_training:
                record.logging("- use char lstm embedding")
            # to be continue....

        # handcraft features
        if opts.use_handcraft:
            if is_training:
                record.logging("- use handcraft features after LSTM layer")
            self._handcrafts = tf.placeholder(
                dtype, [batch_size, len_seq, opts.dim_handcraft])
            scores = tf.concat(2, [scores, tf.tanh(self._handcrafts)])
        else:
            self._handcrafts = tf.Variable(0, trainable=False)

        # dropout before decoding
        if is_training and opts.keep_prob < 1:
            record.logging("- use dropout before decoding")
            scores = tf.nn.dropout(scores, opts.keep_prob)

        # decoding
        if opts.use_crf:    # decoding with crf
            if is_training:
                record.logging("- use CRF layer for decoding")
            scores, self._transition_params = CRF_layer(
                scores, batch_size, len_seq, num_tags, dtype=dtype)
            pred, scores = CRF_decode_tf(scores, self._transition_params)
            loss = CRF_loss(scores, self._targets, self._transition_params)
        else:               # decoding with softmax
            scores = tf.nn.softmax(hidden_layer(
                scores, num_tags, bias=True, name="decode", dtype=dtype))
            loss = seq_loss(scores, self._targets, dtype=dtype)

        # 损失函数, 预测函数等等
        self._scores = scores
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        self.model_path = opts.model_path
        self.saver = tf.train.Saver()

        if not is_training:
            self._train_op = tf.no_op()
            return

        record.logging("- use Gradient Descent as optimizer")
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          opts.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[],
                                      name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        if opts.use_crf:
            self._new_transition_params = tf.placeholder(
                tf.float32, [num_tags, num_tags], name="new_transition_params")
            self._transition_params_update = tf.assign(
                self._transition_params, self._new_transition_params)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_trans_params(self, session, trans_a):
        session.run(self._transition_params_update,
                    feed_dict={self._new_transition_params: trans_a})

    def save(self, sess, name="model"):
        save_path = self.saver.save(sess, os.path.join(self.model_path, name))
        record.logging("Model saved in file: %s" % save_path)

    def restore(self, sess, file):
        self.saver.restore(sess, file)
        record.logging("Model restored from file: %s" % file)

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def input_chars(self):
        return self._input_chars

    @property
    def handcrafts(self):
        return self._handcrafts

    @property
    def scores(self):
        return self._scores

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def initial_forward_state(self):
        return self._initial_forward_state

    @property
    def initial_backward_state(self):
        return self._initial_backward_state

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
