#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import figet


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.input_size = args.context_input_size
        self.rnn_size = args.context_rnn_size
        self.num_directions = args.context_num_directions
        self.num_layers = args.context_num_layers
        assert self.rnn_size % self.num_directions == 0
        self.hidden_size = self.rnn_size // self.num_directions
        super(ContextEncoder, self).__init__()
        self.rnn = nn.LSTM(self.input_size, self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=args.dropout,
                           bidirectional=(self.num_directions == 2))

    def forward(self, input, word_lut, hidden=None):
        indices = None
        if isinstance(input, tuple):
            input, lengths, indices = input

        emb = word_lut(input) # seq_len x batch x emb
        emb = emb.transpose(0, 1)

        if indices is not None:
            emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(emb, hidden)
        if indices is not None:
            outputs = unpack(outputs)[0]
            outputs = outputs[:,indices, :]
        return outputs, hidden_t


class DocEncoder(nn.Module):

    def __init__(self, args):
        self.args = args
        super(DocEncoder, self).__init__()
        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)
        self.W = nn.Linear(args.doc_input_size, args.doc_hidden_size)
        self.U = nn.Linear(args.doc_hidden_size, args.doc_output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.args.dropout:
            return self.relu(self.U(self.tanh(self.W(self.dropout(input)))))
        return self.relu(self.U(self.tanh(self.W(input))))


class Attention(nn.Module):

    def __init__(self, args):
        self.args = args
        self.rnn_size = args.context_rnn_size
        self.attn_size = args.attn_size
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(args.context_input_size, args.context_rnn_size)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, mention, context, mask=None):
        # return self.average(context)
        return self.att_func1(mention, context, mask)

    def average(self, context):
        return torch.mean(context.transpose(0, 1), 1), None

    def att_func1(self, mention, context, mask):
        context = context.transpose(0, 1).contiguous()
        targetT = self.linear_in(mention).unsqueeze(2)   # batch x attn_size x 1
        attn = torch.bmm(context, targetT).squeeze(2)
        if False: # mask is not None:
            attn.data.masked_fill_(mask, -float(1000))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1)) # batch x 1 x seq_len*2
        weighted_context_vec = torch.bmm(attn3, context).squeeze(1)

        if self.args.single_context == 1:
            context_output = weighted_context_vec
        else:
            context_output = self.tanh(weighted_context_vec)

        return context_output, attn


class Classifier(nn.Module):

    def __init__(self, args, vocab):
        self.args = args
        self.vocab = vocab
        self.num_types = vocab.size()
        self.input_size = args.context_rnn_size + args.context_input_size
        if args.use_doc == 1:
            self.input_size += args.doc_output_size
        if args.use_manual_feature == 1:
            self.input_size += args.feature_emb_size
        super(Classifier, self).__init__()
        if args.use_hierarchy == 1:
            self.prior = self.create_prior()
            self.V = torch.nn.Parameter(
                torch.randn(self.num_types, self.input_size).uniform_(
                    -args.param_init, args.param_init
                ), requires_grad=True
            )
        else:
            self.W = nn.Linear(self.input_size, self.num_types, bias=args.bias==1)
        self.sg = nn.Sigmoid()
        self.loss_func = nn.BCEWithLogitsLoss()

    def create_prior(self):
        types = list(self.vocab.label2idx.keys())
        W = torch.zeros((len(types), len(types)))
        for idx in range(self.num_types):
            type_ = self.vocab.get_label(idx)
            assert type_ is not None
            fields = type_.split("/")[1:]
            subtypes = ["/" + "/".join(fields[:i+1]) for i in range(len(fields))]
            for subtype in subtypes:
                sub_idx = self.vocab.lookup(subtype)
                assert sub_idx is not None
                W[idx][sub_idx] = 1
        W = torch.autograd.Variable(W, requires_grad=False).cuda()
        return W

    def forward(self, input, type_vec=None):
        if self.args.use_hierarchy == 1:
            W = torch.matmul(self.prior, self.V).transpose(0, 1)
            logit = torch.matmul(input, W)
        else:
            logit = self.W(input)
        distribution = self.sg(logit)
        loss = None
        if type_vec is not None:
            loss = self.loss_func(logit, type_vec)
        return loss, distribution


class MentionEncoder(nn.Module):

    def __init__(self, args):
        self.dropout = None
        super(MentionEncoder, self).__init__()
        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)

    def forward(self, input, word_lut):
        if self.dropout:
            return self.dropout(input)
        return input


class Model(nn.Module):

    def __init__(self, args, vocabs):
        self.args = args
        super(Model, self).__init__()
        self.word_lut = nn.Embedding(
            vocabs["token"].size(), args.context_input_size,
            padding_idx=figet.Constants.PAD
        )
        if args.use_manual_feature == 1:
            self.feature_lut = nn.Embedding(
                vocabs["feature"].size(), args.feature_emb_size,
                padding_idx=figet.Constants.PAD
            )
        else:
            self.feature_lut = None
        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = None
        self.mention_encoder = MentionEncoder(args)
        self.prev_context_encoder = ContextEncoder(args)
        self.next_context_encoder = ContextEncoder(args)
        if args.use_doc == 1:
            self.doc_encoder = DocEncoder(args)
        self.attention = Attention(args)
        self.classifier = Classifier(args, vocabs["type"])

    def init_params(self, word2vec=False):
        if self.args.use_manual_feature == 1:
            self.feature_lut.weight.data.uniform_(
                -self.args.param_init, self.args.param_init)
        if word2vec:
            pretrained = torch.load(word2vec)
            self.word_lut.weight.data.copy_(pretrained)
            self.word_lut.weight.requires_grad = False

    def forward(self, input):
        mention = input[0]
        prev_context, prev_mask = input[1]
        next_context, next_mask = input[2]
        type_vec = input[3]
        feature = input[4]
        doc = input[5]
        attn = None
        mention_vec = self.mention_encoder(mention, self.word_lut)
        context_vec, attn = self.encode_context(
            prev_context, prev_mask,
            next_context, next_mask,
            mention_vec)
        vecs = [mention_vec, context_vec]
        if self.args.use_doc == 1:
            doc_vec = self.doc_encoder(doc)
            vecs += [doc_vec]
        if feature is not None and self.feature_lut is not None:
            feature_vec = torch.mean(self.feature_lut(feature), 1)
            if self.dropout:
                feature_vec = self.dropout(feature_vec)
            vecs += [feature_vec]
        input_vec = torch.cat(vecs, dim=1)
        loss, distribution = self.classifier(input_vec, type_vec)
        return loss, distribution, attn

    def encode_context(self, *args):
        return self.draw_attention(*args)

    def draw_attention(self, prev_context_vec, prev_mask,
                       next_context_vec, next_mask, mention_vec):
        mask = None
        if self.args.single_context == 1:
            context_vec, _ = self.prev_context_encoder(prev_context_vec, self.word_lut)
        else:
            prev_context_vec, _ = self.prev_context_encoder(prev_context_vec, self.word_lut)
            next_context_vec, _ = self.next_context_encoder(next_context_vec, self.word_lut)
            if False:  # prev_mask is not None and next_mask is not None:
                mask = torch.cat((prev_mask, next_mask), dim=1)
            context_vec = torch.cat((prev_context_vec, next_context_vec), dim=0)
        weighted_context_vec, attn = self.attention(mention_vec, context_vec, mask)
        return weighted_context_vec, attn
