#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import torch
import figet


class Mention(object):

    def __init__(self, line, doc_vec=None):
        self.line = line
        self.doc_vec = doc_vec

    def preprocess(self, vocabs, word2vec, args):
        self.vocabs = vocabs
        self.word2vec = word2vec
        self.context_length = args.context_length
        fields = self.line.split("\t")
        self.start, self.end = int(fields[0]), int(fields[1])
        self.tokens, self.types, self.features = fields[2].split(), fields[3].split(), fields[4].split()
        self.types = self.type_idx()
        self.features = self.feature_idx()
        self.mention = self.mention_idx()
        if args.single_context == 1:
            self.context = self.context_idx()
        else:
            self.prev_context = self.prev_context_idx()
            self.next_context = self.next_context_idx()
        if self.doc_vec is not None:
            self.doc_vec = torch.from_numpy(self.doc_vec)
        self.tokens = None

    def mention_idx(self):
        return torch.mean(torch.stack(
            [self.word2vec[self.vocabs["token"].lookup(token, figet.Constants.UNK)]
             for token in self.tokens[self.start: self.end]]), dim=0).squeeze(0)

    def context_idx(self):
        context = (self.tokens[max(0, self.start - self.context_length): self.start] +
                   [figet.Constants.PAD_WORD] +
                   self.tokens[self.end: min(len(self.tokens), self.end + self.context_length)])
        return self.vocabs["token"].convert_to_idx(context, figet.Constants.UNK_WORD)

    def prev_context_idx(self):
        prev_context = self.tokens[max(0, self.start - self.context_length): self.start]
        return self.vocabs["token"].convert_to_idx(prev_context, figet.Constants.UNK_WORD)

    def next_context_idx(self):
        next_context = self.tokens[self.end: min(len(self.tokens), self.end + self.context_length)]
        return self.vocabs["token"].convert_to_idx(next_context, figet.Constants.UNK_WORD)

    def feature_idx(self):
        return self.vocabs["feature"].convert_to_idx(self.features, figet.Constants.UNK_WORD)

    def type_idx(self):
        type_vec = torch.Tensor(self.vocabs["type"].size()).zero_()
        for type_ in self.types:
            type_idx = self.vocabs["type"].lookup(type_)
            type_vec[type_idx] = 1
        return type_vec
