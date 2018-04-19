#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import torch

import preprocess
import figet
from figet.context_modules.doc2vec import Doc2Vec


def interpret_attention(tokens, start, end, attn, args):
    sent = []
    mention = tokens[start: end]
    for token in tokens[:max(0, start - args.context_length)]:
        sent.append("%s:%.2f" %(token, 0))
    if args.single_context == 1:
        context = (tokens[max(0, start - args.context_length): start] +
                   [figet.Constants.PAD_WORD] +
                   tokens[end: min(len(tokens), end + args.context_length)])
        for i, token in enumerate(context):
            if token == figet.Constants.PAD_WORD:
                sent += ["%s:%.2f" %(x, -1) for x in mention]
            else:
                sent.append("%s:%.2f" %(token, attn[i]*100))
    else:
        prev_context = tokens[max(0, start - args.context_length): start]
        next_context = tokens[end: min(len(tokens), end + args.context_length)]
        for i, token in enumerate(prev_context):
            sent.append("%s:%.2f" %(token, attn[i]*100))
        sent += ["%s:%.2f" %(x, -1) for x in mention]
        for i, token in enumerate(next_context):
            sent.append("%s:%.2f" %(token, attn[-i-1]*100))
    for token in tokens[min(len(tokens), end + args.context_length):]:
        sent.append("%s:%.2f" %(token, 0))
    return " ".join(sent)


def dump_results(type_vocab, lines, preds, attns, args):
    ret = []
    if len(attns) == 0:
        attns = [None]*len(lines)
    for line, (gold_type, pred_type), attn in zip(lines, preds, attns):
        # Get types.
        gold_type = list(sorted(map(type_vocab.get_label, gold_type)))
        pred_type = list(sorted(map(type_vocab.get_label, pred_type)))
        start, end, sent, types = line.split("\t")[:4]
        ref_type = list(sorted(types.split()))
        # assert gold_type == ref_type, " ".join(gold_type) + " <=> " + " ".join(ref_type) + "\n" + line
        # Get attention.
        if attn is not None:
            sent = interpret_attention(sent.split(), int(start), int(end), attn, args)
        ret.append(
            "\t".join(
                [start, end, sent,
                 " ".join(ref_type) + " <=> " + " ".join(pred_type)
                 ]))
    with open(args.pred, "w") as fp:
        fp.write("\n".join(ret))


def read_data(data_file):
    lines = []
    for line in open(data_file):
        line = line.strip()
        fields = line.split("\t")
        if len(fields) not in {5, 8}:
            continue

        start_idx, end_idx = int(fields[0]), int(fields[1])
        tokens = fields[2].split()
        if len(tokens[start_idx: end_idx]) == 0:
            continue
        lines.append(line)
    return lines


def main(args, log):
    # Load checkpoint.
    checkpoint = torch.load(args.save_model)
    vocabs, word2vec, states = checkpoint["vocabs"], checkpoint["word2vec"], checkpoint["states"]
    try:
        idx2threshold = torch.load(args.save_idx2threshold)
    except:
        idx2threshold = None
    log.info("Loaded checkpoint model from %s." %(args.save_model))

    # Load the pretrained model.
    model = figet.Models.Model(args, vocabs)
    model.load_state_dict(states)
    if len(args.gpus) >= 1:
        model.cuda()
    doc2vec = None
    if args.use_doc == 1:
        doc2vec = Doc2Vec(save_path=args.doc2vec_model)
        doc2vec.load()
    log.info("Init the model.")

    # Load data.
    lines = read_data(args.data)
    data = preprocess.make_data(args.data, vocabs, args, doc2vec)
    for mention in data:
        mention.preprocess(vocabs, word2vec, args)
    data = figet.Dataset(data, len(data), args, True)
    log.info("Loaded the data from %s." %(args.data))

    # Inference.
    preds, types, dists, attns = [], [], [], []
    model.eval()
    for i in range(len(data)):
        batch = data[i]
        loss, dist, attn = model(batch)
        preds += figet.adaptive_thres.predict(dist.data, batch[3].data, idx2threshold)
        types += [batch[3].data]
        dists += [dist.data]
        if attn is not None:
            attns += [attn.data]
    types = torch.cat(types, 0).cpu().numpy()
    dists = torch.cat(dists, 0).cpu().numpy()
    if len(attns) != 0:
        attns = torch.cat(attns, 0).cpu().numpy()
    log.info("Finished inference.")

    # Results.
    log.info("| Inference acc. %s |" % (figet.evaluate.evaluate(preds)))
    dump_results(vocabs["type"], lines, preds, attns, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("infer.py")

    # Data options
    parser.add_argument("--data", required=True, type=str,
                        help="Data path.")
    parser.add_argument("--pred", required=True, type=str,
                        help="Prediction output.")
    parser.add_argument("--save_model", default="./save/model.pt", type=str,
                        help="Save the model.")
    parser.add_argument("--save_idx2threshold", default="./save/threshold.pt",
                        type=str, help="Save the model.")

    # Sentence-level context parameters
    parser.add_argument("--context_length", default=10, type=int,
                        help="Max length of the left/right context.")
    parser.add_argument("--context_input_size", default=300, type=int,
                        help="Input size of CotextEncoder.")
    parser.add_argument("--context_rnn_size", default=200, type=int,
                        help="RNN size of ContextEncoder.")
    parser.add_argument("--context_num_layers", default=1, type=int,
                        help="Number of layers of ContextEncoder.")
    parser.add_argument("--context_num_directions", default=2, choices=[1, 2], type=int,
                        help="Number of directions for ContextEncoder RNN.")
    parser.add_argument("--attn_size", default=100, type=int,
                        help=("Attention vector size."))
    parser.add_argument("--single_context", default=0, type=int,
                        help="Use single context.")

    # Manual feature parameters
    parser.add_argument("--use_hierarchy", default=0, type=int,
                        help="Use hierarchy.")
    parser.add_argument("--use_manual_feature", default=0, type=int,
                        help="Use manual feature")
    parser.add_argument("--feature_emb_size", default=50, type=int,
                        help="Feature embedding size.")

    # Doc-level context parameters
    parser.add_argument("--use_doc", default=0, type=int,
                        help="Use doc-level contexts.")
    parser.add_argument("--doc_input_size", default=50, type=int,
                        help="Input size of DocEncoder.")
    parser.add_argument("--doc_hidden_size", default=70, type=int,
                        help="Hidden size of DocEncoder.")
    parser.add_argument("--doc_output_size", default=50, type=int,
                        help="Output size of DocEncoder.")

    # Other parameters
    parser.add_argument("--bias", default=0, type=int,
                        help="Whether to use bias in the linear transformation.")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="Dropout rate for all applicable modules.")
    parser.add_argument('--seed', type=int, default=3435,
                        help="Random seed")
    parser.add_argument('--shuffle', action="store_true",
                        help="Shuffle data.")

    # GPU
    parser.add_argument("--gpus", default=[], nargs="+", type=int,
                        help="Use CUDA on the listed devices.")

    args = parser.parse_args()


    if args.gpus:
        torch.cuda.set_device(args.gpus[0])

    figet.utils.set_seed(args.seed)
    log = figet.utils.get_logging()
    log.debug(args)

    main(args, log)
