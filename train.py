#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import torch

import figet

parser = argparse.ArgumentParser("train.py")

# Data options
parser.add_argument("--data", required=True, type=str,
                    help="Data path.")
parser.add_argument("--save_tuning", default="./save/tuning.pt", type=str,
                    help="Save the intermediate results for tuning.")
parser.add_argument("--save_model", default="./save/model.pt", type=str,
                    help="Save the model.")

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
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="Starting learning rate.")
parser.add_argument("--param_init", default=0.01, type=float,
                    help=("Parameters are initialized over uniform distribution"
                    "with support (-param_init, param_init)"))
parser.add_argument("--batch_size", default=1000, type=int,
                    help="Batch size.")
parser.add_argument("--dropout", default=0.5, type=float,
                    help="Dropout rate for all applicable modules.")
parser.add_argument("--niter", default=150, type=int,
                    help="Number of iterations per epoch.")
parser.add_argument("--epochs", default=15, type=int,
                    help="Number of training epochs.")
parser.add_argument("--max_grad_norm", default=-1, type=float,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("--extra_shuffle", default=1, type=int,
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('--seed', type=int, default=3435,
                    help="Random seed")

# Pretrained word vectors
parser.add_argument("--word2vec", default=None, type=str,
                    help="Pretrained word vectors.")

# GPU
parser.add_argument("--gpus", default=[], nargs="+", type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('--log_interval', type=int, default=50,
                    help="Print stats at this interval.")


args = parser.parse_args()


if args.gpus:
    torch.cuda.set_device(args.gpus[0])

figet.utils.set_seed(args.seed)

log = figet.utils.get_logging()
log.debug(args)


def main():
    # Load data.
    log.debug("Loading data from '%s'." % args.data)

    data = torch.load(args.data)
    vocabs = data["vocabs"]
    word2vec = torch.load(args.word2vec)

    log.info("Loaded data.")

    for split in ("train", "dev", "test"):
        for mention in data[split]:
            mention.preprocess(vocabs, word2vec, args)
    train_data = figet.Dataset(data["train"], args.batch_size, args)
    dev_data = figet.Dataset(data["dev"], len(data["dev"]), args, True)
    test_data = figet.Dataset(data["test"], len(data["test"]), args, True)

    log.info("Loaded datasets.")

    # Build model.
    log.debug("Building model...")

    model = figet.Models.Model(args, vocabs)
    optim = figet.Optim(args.learning_rate, args.max_grad_norm)

    if len(args.gpus) >= 1:
        model.cuda()

    # for p in model.parameters():
    #     p.data.uniform_(-args.param_init, args.param_init)

    model.init_params(args.word2vec)

    optim.set_parameters(filter(lambda p: p.requires_grad, model.parameters()))

    nParams = sum([p.nelement() for p in model.parameters()])
    log.debug("* number of parameters: %d" % nParams)

    coach = figet.Coach(model, vocabs, train_data, dev_data, test_data, optim, args)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    tuning = {
        "type_vocab": vocabs["type"],
        "dev_dist": ret[3],
        "dev_type": ret[4],
        "test_dist": ret[5],
        "test_type": ret[6],
        "test_raw_data": ret[7]
    }
    checkpoint = {
        "vocabs": vocabs,
        "word2vec": word2vec,
        "states": ret[2]
    }
    torch.save(tuning, args.save_tuning)
    torch.save(checkpoint, args.save_model)

    log.info("Done!")


if __name__ == "__main__":
    main()
