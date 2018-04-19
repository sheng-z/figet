#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import argparse
from functools import partial
import numpy as np
import multiprocessing as mp
import torch
import figet
from figet import utils

log=utils.get_logging()


def tune(baseline, dist, type_, num_types, init_threshold):
    idx2threshold = {idx: init_threshold for idx in xrange(num_types)}
    pool = mp.Pool(processes=8)
    func = partial(search_threshold,
                   init_threshold=init_threshold,
                   num_types=num_types,
                   dist=dist,
                   type_=type_,
                   baseline=baseline)
    for idx, best_t in pool.map(func, xrange(num_types)):
        idx2threshold[idx] = best_t
    return idx2threshold


def search_threshold(idx, init_threshold, num_types, dist, type_, baseline):
    # Search the best thresholds.
    idx2threshold = {i: init_threshold for i in xrange(num_types)}
    best_t = idx2threshold[idx]
    for t in list(np.linspace(0, 1.0, num=20)):
        idx2threshold[idx] = t
        pred = predict(dist, type_, idx2threshold)
        _, _, temp = figet.evaluate.strict(pred)
        if temp > baseline:
            best_t = t
    print ('-', end='')
    return idx, best_t


def predict(pred_dist, Y, idx2threshold=None):
    ret = []
    batch_size = pred_dist.shape[0]
    for i in xrange(batch_size):
        dist = pred_dist[i]
        type_vec = Y[i]
        pred_type = []
        gold_type = []
        for idx, score in enumerate(list(type_vec)):
            if score > 0:
                gold_type.append(idx)
        midx, score = max(enumerate(list(dist)), key=lambda x: x[1])
        pred_type.append(midx)
        for idx, score in enumerate(list(dist)):
            if idx2threshold is None:
                threshold = 0.5
            else:
                threshold = idx2threshold[idx]
            if score > threshold and idx != midx:
                pred_type.append(idx)
        ret.append([gold_type, pred_type])
    return ret


def main(args):
    data = torch.load(args.data)
    type_vocab = data["type_vocab"]

    # Baseline predictions.
    dev_predictions = predict(data["dev_dist"], data["dev_type"])
    test_predictions = predict(data["test_dist"], data["test_type"])

    log.info("| Baseline | dev acc. %s | test acc. %s |" % (
        figet.evaluate.evaluate(dev_predictions),
        figet.evaluate.evaluate(test_predictions)))

    # Baseline on dev.
    _, _, baseline = figet.evaluate.strict(dev_predictions)

    idx2threshold = tune(baseline, data["dev_dist"], data["dev_type"],
                         type_vocab.size(), args.init_threshold)

    torch.save(idx2threshold, args.optimal_thresholds)
    print ('')
    # After tuning.
    dev_predictions = predict(data["dev_dist"], data["dev_type"], idx2threshold)
    test_predictions = predict(data["test_dist"], data["test_type"], idx2threshold)

    log.info("| Tuned | dev acc. %s | test acc. %s |" % (
        figet.evaluate.evaluate(dev_predictions),
        figet.evaluate.evaluate(test_predictions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("adaptive_thres.py")
    parser.add_argument("--data", help="The data for tuning.")
    parser.add_argument("--init_threshold", default=0.5, type=float,
                        help="The init threshold.")
    parser.add_argument("--optimal_thresholds", default="./save/threshold.pt",
                        help="The optimal threshold.")
    args = parser.parse_args()
    main(args)
