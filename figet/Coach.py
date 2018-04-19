#!/usr/bin/env python
# encoding: utf-8

import time
import copy
import numpy as np
from tqdm import tqdm
import torch

import figet

log = figet.utils.get_logging()


class Coach(object):

    def __init__(self, model, vocabs, train_data, dev_data, test_data, optim, args):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.optim = optim
        self.args = args

    def train(self):
        log.debug(self.model)

        self.start_time = time.time()
        # Early stopping.
        best_dev_f1, best_epoch, best_state = None, None, None
        # Adaptive threshods.
        best_dev_dist, dev_labels = None, None
        test_dist, test_labels, raw_test_data = None, None, None
        # Run epochs.
        for epoch in xrange(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            # Record the best results on dev.
            dev_results = self.validate()
            test_results = self.validate(self.test_data)
            _, _, dev_f1 = figet.evaluate.strict(dev_results[1])
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_dev_dist, dev_labels = dev_results[2:4]
                test_dist, test_labels, raw_test_data = test_results[2:5]
                log.info("* the new best dev f1: %.2f" %(best_dev_f1*100))
            log.info(
                "| epoch %d | dev acc. %s | test acc. %s | loss (%.2f, %.2f, %.2f) |"
                % (epoch, figet.evaluate.evaluate(dev_results[1]),
                   figet.evaluate.evaluate(test_results[1]),
                   train_loss*100, dev_results[0]*100, test_results[0]*100))

        return (best_dev_f1, best_epoch, best_state,
                best_dev_dist, dev_labels,
                test_dist, test_labels, raw_test_data)

    def train_epoch(self, epoch):
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)
        batch_order = list(xrange(niter)) # torch.randperm(niter)

        total_tokens, report_tokens = 0, 0
        total_loss, report_loss = [], []
        start_time = time.time()
        self.model.train()
        for i in tqdm(xrange(niter), desc="train_one_epoch"):
            batch_idx = batch_order[i]
            batch = self.train_data[batch_idx]

            self.model.zero_grad()
            loss, _, _ = self.model(batch)
            self.optim.optimizer.zero_grad()
            loss.backward()
            self.optim.step()

            # Stats.
            prev_context, next_context = batch[1], batch[2]
            if isinstance(prev_context, tuple):
                num_tokens = prev_context[0][0].data.ne(figet.Constants.PAD).sum()
            else:
                num_tokens = prev_context.data.ne(figet.Constants.PAD).sum()
            if self.args.single_context == 0:
                if isinstance(next_context, tuple):
                    num_tokens = next_context[0][0].data.ne(figet.Constants.PAD).sum()
                else:
                    num_tokens = next_context.data.ne(figet.Constants.PAD).sum()
            total_tokens += num_tokens
            report_tokens += num_tokens
            total_loss += [loss.data[0]]
            report_loss += [loss.data[0]]
            if i % self.args.log_interval == -1 % self.args.log_interval:
                log.debug(
                    "Epoch %2d | %5d/%5d | loss %6.2f | %3.0f ctx tok/s | %6.0f s elapsed"
                    %(epoch, i+1, len(self.train_data), np.mean(report_loss),
                      report_tokens/(time.time()-start_time),
                      time.time()-self.start_time))
                report_tokens = 0
                report_loss = []
                start_time = time.time()

        return np.mean(total_loss)

    def validate(self, data=None):
        data = data if data is not None else self.dev_data
        total_loss = []
        self.model.eval()
        predictions = []
        dists, labels, raw_data = [], [], []
        for i in xrange(len(data)):
            batch = data[i]
            loss, dist, attn = self.model(batch)
            predictions += figet.adaptive_thres.predict(dist.data, batch[3].data)
            dists += [dist.data]
            labels += [batch[3].data]
            raw_data += [mention.line for mention in batch[-1]]
            total_loss += [loss.data[0]]
        dists = torch.cat(dists, 0)
        labels = torch.cat(labels, 0)
        return (np.mean(total_loss), predictions,
                dists.cpu().numpy(), labels.cpu().numpy(), raw_data)
