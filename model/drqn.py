import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class DRQN(Chain):
    def __init__(self, input_num, action_num):
        print("DRQN Model", input_num, action_num)
        super(DRQN, self).__init__(
            fc1=L.Linear(input_num, 256),
            lstm=L.LSTM(256, 256),
            fc2=L.Linear(256, action_num),
        )

    def q_function(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.lstm(h))
        h = self.fc2(h)
        return h


    def is_reccurent(self):
        return True

    def reset_state(self):
        self.lstm.reset_state()


    def push_state(self):
        self.c = self.lstm.c
        self.h = self.lstm.h
        self.lstm.reset_state()

    def pop_state(self):
        self.lstm.set_state(c=self.c, h=self.h)
