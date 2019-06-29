# *-* coding: utf-8

import os
import time
import codecs


RECORD_FILE = '../eval/record.txt'

loggin_infos = []


def format_time(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def format_duration(t):
    ret = ""
    if t >= 3600:
        ret = "%d hour(s) " % (t // 3600)
        t = t % 3600
    if t >= 60:
        ret += "%d min(s) " % (t // 60)
        t = t % 60
    ret += "%d second(s)" % t
    return ret


def logging(info):
    print(info)
    loggin_infos.append("logging at {}: {}".format(format_time(time.time()), info))


def record(opts, valid_score, test_score, start_time):
    to_write = ['*' * 80]
    cur_time = time.time()
    to_write.append("{:20s}: {}, {}".format("scores(v/t)", valid_score, test_score))
    to_write.append("{:20s}: {}".format("start at", format_time(start_time)))
    to_write.append("{:20s}: {}".format("end at", format_time(cur_time)))
    to_write.append("{:20s}: {}".format("run time", format_duration(cur_time - start_time)))
    for item in opts.__dict__:
        to_write.append("{:20s}: {}".format(item, opts.__dict__[item]))

    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), RECORD_FILE)
    with codecs.open(file, 'a', 'utf-8') as fout:
        fout.write('\n'.join(to_write))
        if loggin_infos:
            fout.write('\n')
            fout.write('\n'.join(loggin_infos))
        fout.write('\n\n')
    print("results saved to", file)
