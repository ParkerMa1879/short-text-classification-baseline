import numpy as np
import os
import logging
import gensim

from tweet_classification import BatchFeeder, Process
from data.util import data_set
from sklearn.linear_model import LogisticRegression


def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


if __name__ == '__main__':
    _log = create_log("./log/bench_mark.log")
    # load data
    data = data_set()
    x, y = data["sentence"], data["label"]
    ratio = float(np.sum(y == 0) / len(y))
    _log.info("Balance of positive label in training data: %0.3f" % ratio)
    w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)
    pre_process = Process("embed_avg", {"dim": w2v.vector_size, "model": w2v, "path": "./data/random_dict.json"})

    # preprocess
    _log.info("Fixed validation (accuracy for balanced data)")
    feeder = BatchFeeder(x, y, 6089, validation=0.05, process=pre_process, fix_validation=False)
    x, y = feeder.next()
    x_v, y_v = feeder.next_valid()
    print(x.shape, y.shape)
    print(x_v.shape, y_v.shape)

    # train
    # clf = svm.SVC(verbose=False, shrinking=False)
    # clf.fit(x, y)
    # acc = np.sum(y_v == clf.predict(x_v))/len(y_v)
    # _log.info("SVM-accuracy: %0.3f" % acc)

    clf = LogisticRegression(verbose=False)
    clf.fit(x, y)
    acc = np.sum(y_v == clf.predict(x_v))/len(y_v)

    _log.info("LR-accuracy: %0.3f" % acc)

    feeder.finalize()
