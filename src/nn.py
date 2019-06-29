# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def hidden_layer(inputs, dim_output, keep_prob=1.0,
                 name="hidden", bias=True, dtype=tf.float32):
    """线性连接层, no activating function
    scores = inputs * weight + bias
    """
    batch_size = int(inputs.get_shape()[0]) or inputs.shape[0]
    len_seq = int(inputs.get_shape()[1]) or inputs.shape[1]
    dim_input = int(inputs.get_shape()[2]) or inputs.shape[2]
    input_2d = tf.reshape(inputs, [-1, dim_input])

    hidden_w = tf.get_variable(
        name + "_weight", [dim_input, dim_output], dtype=dtype)
    scores = tf.matmul(input_2d, hidden_w)
    if bias:
        hidden_b = tf.get_variable(
            name + "_bias", [dim_output], dtype=dtype)
        scores = scores + hidden_b
    if keep_prob < 1.0:
        scores = tf.nn.dropout(scores, keep_prob)

    scores = tf.reshape(scores, [batch_size, len_seq, dim_output])
    return scores


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    w = tf.get_variable(
        name + '_weight', [k_h, k_w, input_.get_shape()[-1], output_dim])
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
    return conv


def CNN_layer(inputs, dim_output=50):
    """CNNs for char-level embedding
    Args:
        inputs: batchsize * len_char * dim_char
        dim_output: specify dimension of output embeddings

    Return:
        output: batch_size * dim_output
    """
    len_char = int(inputs.get_shape()[1]) or inputs.shape[1]
    dim_char = int(inputs.get_shape()[2]) or inputs.shape[2]

    pools = []
    for idx in range(3, 4):
        conv = conv2d(inputs, dim_output, idx, dim_char, name="conv" + str(idx))
        pool = tf.squeeze(tf.nn.max_pool(
            tf.tanh(conv), [1, len_char - idx + 1, 1, 1], [1, 1, 1, 1], 'VALID'))
        pools.append(pool)
    output = tf.concat(1, pools)
    dim_in = output.get_shape()[1]
    concat_weight = tf.get_variable("char_embed_concat", [dim_in, dim_output])
    output = tf.matmul(output, concat_weight)
    return output


def get_LSTM_cell(dim, keep_prob=1.0, cell_type="BasicLSTMCell"):
    if cell_type == 'BasicLSTMCell':
        cell = tf.nn.rnn_cell.BasicLSTMCell(dim, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.GRUCell(dim)

    if keep_prob < 1.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


def bi_LSTM_layer(inputs, lstm_cell, num_layers, dtype=tf.float32):
    """双向LSTM神经网络, 参考Lample2016论文
    一个正向LSTM顺序处理时序输入
    一个反向LSTM逆序处理时序输入
    一个线性连接层联合正向LSTM和反向LSTM的结果

    实现方法参考tensorflow RNN 教程.

    Args:
        inputs: 双向神经网络的输入, 规模为[batch_size, len_seq, dim_embed)]
            * batch_size: batch规模
            * len_seq: LSTM处理的时序长度, 也即处理一个句子的长度
            * dim_embed: 神经元维度
        num_layers: 正向/反向LSTM网络的层数
        keep_prob: dropout相关

    Returns:
        initial_forward_state： 正向LSTM初始状态
        initial_backward_state: 反向LSTM初始状态
        scores: 最终双向LSTM连接输出的结果
    """
    batch_size = int(inputs.get_shape()[0]) or inputs.shape[0]
    len_seq = int(inputs.get_shape()[1]) or inputs.shape[1]
    dim_embed = int(inputs.get_shape()[2]) or inputs.shape[2]

    forward_lstm_layer = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell] * num_layers)
    backward_lstm_layer = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell] * num_layers)

    initial_forward_state = forward_lstm_layer.zero_state(
        batch_size, dtype)
    initial_backward_state = forward_lstm_layer.zero_state(
        batch_size, dtype)

    forward_outputs = []
    forward_state = initial_forward_state
    with tf.variable_scope("RNN1"):
        for time_step in range(len_seq):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (foutput, forward_state) = forward_lstm_layer(
                inputs[:, time_step, :], forward_state)
            forward_outputs.append(foutput)

    backward_outputs = []
    backward_state = initial_backward_state
    with tf.variable_scope("RNN2"):
        for time_step in range(len_seq):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (boutput, backward_state) = backward_lstm_layer(
                inputs[:, len_seq - time_step - 1, :], backward_state)
            backward_outputs.append(boutput)

    forward_output = tf.reshape(
        tf.concat(1, forward_outputs), [-1, dim_embed])
    backward_output = tf.reshape(
        tf.concat(1, backward_outputs), [-1, dim_embed])
    output = tf.concat(1, [forward_output,
                           tf.reverse(backward_output, [True, False])])
    output = tf.reshape(output, [batch_size, len_seq, dim_embed * 2])
    return initial_forward_state, initial_backward_state, output


def char_LSTM_layer(chars, dim_output):
    batch_size = int(chars.get_shape()[0]) or chars.shape[0]
    len_seq = int(chars.get_shape()[1]) or chars.shape[1]
    len_char = int(chars.get_shape()[2]) or chars.shape[2]
    dim_char = int(chars.get_shape()[3]) or chars.shape[3]


def CRF_layer(inputs, batch_size, len_seq, num_tags, dtype=tf.float32):
    """CRF层解码 这一层要训练的参数的是状态转移矩阵
    我可能写了一个假的CRF层...
    """
    scores = hidden_layer(inputs, num_tags,
                          bias=False, name="crf", dtype=dtype)
    transition_params = tf.get_variable("transition_params",
                                        [num_tags, num_tags])
    return scores, transition_params


def CRF_decode_tf(scores, transition_params):
    """CRF损失函数"""
    batch_size = int(scores.get_shape()[0]) or scores.shape[0]
    len_seq = int(scores.get_shape()[1]) or scores.shape[1]
    num_tags = int(scores.get_shape()[2]) or scores.shape[2]

    trellis = [tf.squeeze(scores[:, 0, :])]
    backpointers = [tf.zeros([batch_size, num_tags], dtype=tf.int32)]
    for t in range(1, len_seq):
        v = tf.expand_dims(trellis[t - 1], 2) + transition_params
        trellis.append(scores[:, t, :] + tf.reduce_max(v, 1))
        backpointers.append(tf.cast(tf.argmax(v, 1), tf.int32))
    backpointers = tf.transpose(tf.pack(backpointers), [1, 0, 2])

    viterbi = [tf.cast(tf.argmax(trellis[-1], 1), tf.int32)]
    for t in range(len_seq - 2, -1, -1):
        indices = tf.concat(1, [tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1]),
                                tf.fill([batch_size, 1], t),
                                tf.reshape(viterbi[-1], [batch_size, 1])])
        viterbi.append(tf.gather_nd(backpointers, indices))
    trellis = tf.transpose(tf.pack(trellis), [1, 0, 2])
    viterbi = tf.transpose(tf.pack(viterbi), [1, 0])
    return viterbi, trellis


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = tf.reduce_max(x, axis=axis, keep_dims=True)
    xmax_ = tf.reduce_max(x, axis=axis)
    return tf.reduce_sum(xmax_ + tf.log(tf.exp(x - xmax)), axis=axis)


def CRF_loss(scores, targets, transition_params):
    """CRF损失函数"""
    batch_size = int(scores.get_shape()[0]) or scores.shape[0]
    len_seq = int(scores.get_shape()[1]) or scores.shape[1]

    sequence_lengths = np.full(batch_size, len_seq, dtype=np.int32)
    sequence_lengths_t = tf.constant(sequence_lengths)
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
        scores, targets, sequence_lengths_t,
        transition_params=transition_params)
    loss = tf.reduce_mean(-log_likelihood, -1)
    return loss


def trans_scores(scores, targets, transition_params):
    """binary scores"""
    len_seq = int(scores.get_shape()[1])
    num_tags = transition_params.get_shape()[0]
    num_transitions = array_ops.shape(targets)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = array_ops.slice(targets, [0, 0], [-1, num_transitions])
    end_tag_indices = array_ops.slice(targets, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = array_ops.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = array_ops.gather(flattened_transition_params,
                                     flattened_transition_indices)
    binary_scores = math_ops.reduce_logsumexp(binary_scores, reduction_indices=1) / len_seq
    return binary_scores


def CRF_decode(scores, transition_params):
    """维特比算法解码"""
    batch_size = scores.shape[0]
    len_seq = scores.shape[1]
    viterbis = np.empty([batch_size, len_seq])
    for i in range(batch_size):
        viterbi, _ = tf.contrib.crf.viterbi_decode(scores[i].squeeze(),
                                                   transition_params)
        viterbis[i] = viterbi
    return viterbis.astype(int)


def dp_decode(scores, dtags, max_length=5):
    """没有用, 写搓了.

    对长度为len_seq的序列, 复杂度为num_tag * num_tag * len_seq * max_length"""
    batch_size = scores.shape[0]
    len_seq = scores.shape[1]
    all_tag_types = [t[2:] for t in dtags.vocab if len(t) >= 2]
    all_tag_types = list(set(all_tag_types))
    all_tag_types.append('O')
    num_tag_types = len(all_tag_types)

    def whole_tag_score(score, tid):
        if all_tag_types[tid] == 'O':
            return np.sum(score[:, dtags.get_id('O')])
        tag = all_tag_types[tid]
        tag_len = score.shape[0]
        if tag_len == 1:
            return score[0, dtags.get_id('U-' + tag)]
        val = score[0, dtags.get_id('B-' + tag)] + score[-1, dtags.get_id('L-' + tag)]
        if tag_len > 2:
            val += np.sum(score[1:-1, dtags.get_id('I-' + tag)])
        return val

    def pred_from_dp(dp_val, dp_len, dp_last_tag):
        pred = []
        i, j = len_seq - 1, 0
        for k in range(num_tag_types):
            if dp_val[i][k] > dp_val[i][j]:
                j = k
        while i >= 0:
            tag = all_tag_types[j]
            if tag == 'O':
                pred.extend([dtags.get_id('O') for _ in range(dp_len[i][j])])
            elif dp_len[i][j] == 1:
                pred.append(dtags.get_id('U-' + tag))
            else:
                pred.append(dtags.get_id('L-' + tag))
                if dp_len[i][j] > 2:
                    pred.extend([dtags.get_id('I-' + tag) for _ in range(dp_len[i][j] - 2)])
                pred.append(dtags.get_id('B-' + tag))
            old_i = i
            i -= dp_len[i][j]
            j = dp_last_tag[old_i][j]
        pred.reverse()
        return pred

    preds = []
    for b in range(batch_size):
        dp_val = [[0.0 for j in range(len(all_tag_types))] for i in range(len_seq)]
        dp_len = [[1 for j in range(len(all_tag_types))] for i in range(len_seq)]
        dp_last_tag = [[j for j in range(len(all_tag_types))] for i in range(len_seq)]
        for i in range(len_seq):
            for t1 in range(num_tag_types):
                for j in range(max([-1, i - max_length]), i, 1):
                    val_a = whole_tag_score(scores[b, j + 1: i + 1, :], t1)
                    for t2 in range(num_tag_types):
                        if t2 != t1:
                            val = val_a if j == -1 else val_a + dp_val[j][t2]
                            if val > dp_val[i][t1]:
                                dp_val[i][t1] = val
                                dp_len[i][t1] = i - j
                                dp_last_tag[i][t1] = t2
        preds.append(pred_from_dp(dp_val, dp_len, dp_last_tag))
    return np.array(preds)


def seq_loss(scores, targets, dtype=tf.float32):
        """tensorflow-v0.11版中RNN教程PTB模型中的损失函数"""
        batch_size = int(scores.get_shape()[0]) or scores.shape[0]
        len_seq = int(scores.get_shape()[1]) or scores.shape[1]
        scores = tf.reshape(scores, [batch_size * len_seq, -1])
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [scores], [tf.reshape(targets, [-1])],
            [tf.ones([batch_size * len_seq], dtype=dtype)])
        return tf.reduce_sum(tf.reshape(loss, [batch_size, len_seq]), 1)


def test():
    batch_size = 2
    len_seq = 10
    num_tags = 5
    
    scores = np.random.random([batch_size, len_seq, num_tags])
    transition_params = np.random.random([num_tags, num_tags])
    targets = np.random.random([batch_size, len_seq]) * num_tags % num_tags
    scores = scores.astype(np.float32)
    transition_params = transition_params.astype(np.float32)
    targets = targets.astype(np.int32)

    tscores = tf.placeholder(tf.float32, [batch_size, len_seq, num_tags])
    ttransition_params = tf.placeholder(tf.float32, [num_tags, num_tags])
    ttargets = tf.placeholder(tf.int32, [batch_size, len_seq])
    trellis = CRF_decode_tf(tscores, ttransition_params)
    
    sess = tf.Session()
    trellis = sess.run(trellis, feed_dict={tscores: scores, ttransition_params: transition_params, ttargets: targets})
    print(trellis)


if __name__ == "__main__":
    test()
