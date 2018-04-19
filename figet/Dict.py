#!/usr/bin/env python
# encoding: utf-8

import torch

class Dict(object):

    def __init__(self, data=None, lower=False):
        self.idx2label = {}
        self.label2idx = {}
        self.frequencies = {}
        self.lower = lower

        self.special = []

        if data is not None:
            if type(data) == str:
                self.load_file(data)
            else:
                self.add_specials(data)

    def size(self):
        return len(self.idx2label)

    def load_file(self, filepath):
        for line in open(filepath):
            fields = line.strip().split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    def write_file(self, filepath):
        with open(filepath, "w") as f:
            for i in xrange(self.size()):
                label = self.idx2label[i]
                f.write("%s %d\n" % (label, i))

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label2idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx2label[idx]
        except KeyError:
            return default

    def add_special(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    def add_specials(self, labels):
        for label in labels:
            self.add_special(label)

    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idx2label[idx] = label
            self.label2idx[label] = idx
        else:
            if label in self.label2idx:
                idx = self.label2idx[label]
            else:
                idx = len(self.idx2label)
                self.idx2label[idx] = label
                self.label2idx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size=None):
        if size and size >= self.size():
            return self

        if size is None:
            size = self.size()

        freq = torch.Tensor(
                        [self.frequencies[i] for i in xrange(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        ret = Dict()
        ret.lower = self.lower

        for i in self.special:
            ret.add_special(self.idx2label[i])

        for i in idx[:size]:
            ret.add(self.idx2label[i])

        return ret

    def convert_to_idx(self, labels, unk=None, bos=None, eos=None,
                       _type=torch.LongTensor):
        vec = []

        if bos is not None:
            vec += [self.lookup(bos)]

        unk = self.lookup(unk)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eos is not None:
            vec += [self.lookup(eos)]

        return _type(vec)

    def convert_to_labels(self, idx, eos=None):
        labels = []
        if len(idx.size()) == 0:
            return labels
        for i in idx:
            labels += [self.get_label(i)]
            if i == eos:
                break
        return labels

