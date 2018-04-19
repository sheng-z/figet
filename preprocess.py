#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
from tqdm import tqdm
import torch

import figet
from figet.context_modules.doc2vec import Doc2Vec

log = figet.utils.get_logging()


def make_vocabs(args):
    token_vocab = figet.Dict(
        [figet.Constants.PAD_WORD, figet.Constants.UNK_WORD],
        lower=args.lower)
    feature_vocab = figet.Dict([figet.Constants.UNK_WORD])
    type_vocab = figet.Dict()

    all_files = (args.train, args.dev, args.test)
    bar = tqdm(desc="make_vocabs", total=figet.utils.wc(all_files))
    for data_file in all_files:
        for line in open(data_file):
            bar.update()
            fields = line.strip().split("\t")
            tokens, types, features = fields[2].split(), fields[3].split(), fields[4].split()
            for token in tokens:
                token_vocab.add(token)
            for feature in features:
                feature_vocab.add(feature)
            for type_ in types:
                type_vocab.add(type_)
    bar.close()
    token_vocab.prune()
    feature_vocab.prune()
    type_vocab.prune()

    log.info("Created vocabs:\n\t#token: %d\n\t#feature: %d\n\t#type: %d"
          % (token_vocab.size(), feature_vocab.size(), type_vocab.size()))

    return {"token": token_vocab, "feature": feature_vocab, "type": type_vocab}


def make_data(data_file, vocabs, args, doc2vec=None):
    count, ignored = 0, 0
    data, sizes = [], []
    for line in tqdm(open(data_file), total=figet.utils.wc(data_file)):
        line = line.strip()
        fields = line.split("\t")
        if len(fields) not in {5, 8}:
            ignored += 1
            continue

        start_idx, end_idx = int(fields[0]), int(fields[1])
        tokens = fields[2].split()
        if len(tokens[start_idx: end_idx]) == 0:
            ignored += 1
            continue

        doc_vec = None
        if args.use_doc == 1:
            if len(fields) == 5:
                doc = fields[2]
            else:
                doc = fields[7].replace('\\n', ' ').strip()
            doc_vec = doc2vec.transform(doc)
        mention = figet.Mention(line, doc_vec)
        data.append(mention)
        sizes.append(len(tokens))
        count += 1

    if args.shuffle:
        log.info("... shuffling sentences.")
        perm = torch.randperm(len(data))
        data = [data[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

        log.info('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        data = [data[idx] for idx in perm]

    log.info("Prepared %d mentions (%d ignored due to malformed input.)" %(count, ignored))

    return data


def make_word2vec(filepath, vocab):
    token2vec = {}
    log.info("Start loading pretrained word vecs")
    for line in tqdm(open(filepath), total=figet.utils.wc(filepath)):
        fields = line.strip().split()
        token = fields[0]
        vec = list(map(float, fields[1:]))
        token2vec[token] = torch.Tensor(vec)

    ret = []
    oov = 0
    unk_vec = token2vec["unk"]
    for idx in xrange(vocab.size()):
        token = vocab.idx2label[idx]
        if token == figet.Constants.PAD_WORD:
            ret.append(torch.zeros(unk_vec.size()))
            continue
        if token in token2vec:
            vec = token2vec[token]
        else:
            oov += 1
            vec = unk_vec
        ret.append(vec)
    ret = torch.stack(ret)
    log.info("* OOV count: %d" %oov)
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def main(args):

    doc2vec = None
    if args.use_doc == 1:
        doc2vec = Doc2Vec(save_path=args.save_doc2vec)
        doc2vec.load()

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)

    log.info("Preparing training...")
    train = make_data(args.train, vocabs, args, doc2vec)
    log.info("Preparing dev...")
    dev = make_data(args.dev, vocabs, args, doc2vec)
    log.info("Preparing test...")
    test = make_data(args.test, vocabs, args, doc2vec)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs["token"])
    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + ".word2vec"))
    torch.save(word2vec, args.save_data + ".word2vec")


    log.info("Saving data to '%s'..." % (args.save_data + ".data.pt"))
    save_data = {"vocabs": vocabs, "train": train, "dev": dev, "test": test}
    torch.save(save_data, args.save_data + ".data.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--train", required=True,
                        help="Path to the training data.")
    parser.add_argument("--dev", required=True,
                        help="Path to the dev data.")
    parser.add_argument("--test", required=True,
                        help="Path to the test data.")
    parser.add_argument("--word2vec", default="", type=str,
                        help="Path to pretrained word vectors.")

    # Ops
    parser.add_argument("--use_doc", default=0, type=int,
                        help="Whether to use the doc context or not.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle data.")
    parser.add_argument('--seed', type=int, default=3435,
                        help="Random seed")
    parser.add_argument('--lower', action='store_true', help='lowercase data')

    # Output data
    parser.add_argument("--save_data", required=True,
                        help="Path to the output data.")
    parser.add_argument("--save_doc2vec",
                        help="Path to the doc2vec model.")

    args = parser.parse_args()

    figet.utils.set_seed(args.seed)

    main(args)
