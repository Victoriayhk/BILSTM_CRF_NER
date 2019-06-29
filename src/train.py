# *-* coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

import options
import data_loader
import evaluate
import record
from model import NERModel
from nn import CRF_decode
# from nn import dp_decode


flags = tf.flags
logging = tf.logging
flags.DEFINE_string("language", "en", "specify language")
flags.DEFINE_integer("len_char", None, "specity max length of a word")
flags.DEFINE_string("data_path", None, "specify data dataset")
flags.DEFINE_string("restore", None, "restore model from file")
FLAGS = flags.FLAGS


def run_epoch(session, model, data, name, display=0):
    opts = options.opts
    epoch_size = data.data_size(name) // model.batch_size // model.len_seq

    start_time = time.time()
    costs = 0.0
    iters = 0
    forward_state = session.run(model.initial_forward_state)
    backward_state = session.run(model.initial_backward_state)
    xs, ys, preds = [], [], []

    for step, (x, y, c, f) in enumerate(data.iterator(name, opts.batch_size,
                                        opts.len_seq)):
        # feed_dict
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.input_chars] = c
        feed_dict[model.handcrafts] = f
        feed_dict[model.initial_forward_state] = forward_state
        feed_dict[model.initial_backward_state] = backward_state

        # cost & scores & pred
        if opts.use_crf:
            fetches = [model.cost, model.scores,
                       model.transition_params, model.train_op]
            cost, score, transition_params, _ = session.run(fetches, feed_dict)
            pred = CRF_decode(score, transition_params)
        else:
            fetches = [model.cost, model.scores, model.train_op]
            cost, score, _ = session.run(fetches, feed_dict)
            pred = np.argmax(score, -1)
            # pred = dp_decode(score, data.dtags)

        costs += cost
        iters += model.len_seq

        xs.append(x)
        ys.append(y)

        preds.append(pred)
        if display == 1 and step % (epoch_size // 5) == 5:
            print("%.3f perplexity: %10.6f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

    x = evaluate.reshape(xs, opts.batch_size, opts.len_seq)
    y = evaluate.reshape(ys, opts.batch_size, opts.len_seq)
    pred = evaluate.reshape(preds, opts.batch_size, opts.len_seq)
    x = [data.dwords.get_token(i) for i in x]
    y = [data.dtags.get_token(i) for i in y]
    pred = [data.dtags.get_token(i) for i in pred]
    return evaluate.conll_eval(x, y, pred, opts.eval_path)


def train():
    # 配置信息
    options.init(FLAGS)

    # 读入数据
    print("Preparing data...")
    data = data_loader.ConllLoader()
    options.opts.vocab_size = data.vocab_size
    options.opts.num_tags = data.num_tags
    options.opts.dim_handcraft = data.dim_handcraft
    options.opts.char_vocab_size = data.char_vocab_size
    opts = options.opts

    # 输出配置信息
    for item in opts.__dict__:
        print("{:20s}: {}".format(item, opts.__dict__[item]))

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-opts.init_scale,
                                                    opts.init_scale)

        # 建模
        print("\n\nBuilding graphs...")
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = NERModel(data.dwords, is_training=True, dtype=tf.float32)
            if opts.restore:
                m.restore(session, opts.restore)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = NERModel(data.dwords, is_training=False, dtype=tf.float32)
            mtest = NERModel(data.dwords, is_training=False, dtype=tf.float32)

        tf.global_variables_initializer().run()

        best_valid = -np.inf
        best_test = -np.inf
        start_time = time.time()
        print("\n\nRunning epoches...")
        try:
            for i in range(opts.max_max_epoch):
                lr_decay = opts.learning_rate_decay ** max(i - opts.max_epoch, 0.0)
                m.assign_lr(session, opts.learning_rate * lr_decay)     # 学习率

                print("Epoch: %d Learning rate: %f" % (i + 1, session.run(m.lr)))
                run_epoch(session, m, data, "train", display=1)

                print("Validating...")
                valid_score = run_epoch(session, mvalid, data, "valid")
                if valid_score > best_valid:
                    print("New best score on validation dataset:", valid_score)
                    best_valid = valid_score
                    # mvalid.save(session, name="model")

                if (i + 1) % 10 == 0:
                    print("Test...")
                    test_score = run_epoch(session, mtest, data, "test")
                    if test_score > best_test:
                        print("New best score on test dataset:", test_score)
                        best_test = test_score
        except KeyboardInterrupt:
            record.logging("epoches finished = {}".format(i + 1))
            record.record(opts, best_valid, best_test, start_time)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
