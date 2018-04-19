#!/usr/bin/env python
# encoding: utf-8


import sys
import logging
import random
import numpy as np
import torch

def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_logging(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

def wc(files):
    if type(files) is list or type(files) is tuple:
        pass
    else:
        files = [files]
    return sum([sum([1 for _ in open(file)]) for file in files])
