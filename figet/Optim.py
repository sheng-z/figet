#!/usr/bin/env python
# encoding: utf-8

import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def __init__(self, lr, max_grad_norm):
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def set_parameters(self, params):
        self.params = list(params)
        self.optimizer = optim.Adam(self.params, lr=self.lr)

    def step(self):
        if self.max_grad_norm != -1:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict())
