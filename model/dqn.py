import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class DQN(Chain):
    def __init__(self, input_num, action_num):
        print("DQN Model", input_num, action_num)
        super(DQN, self).__init__(
            fc1=L.Linear(input_num, 256),
            fc2=L.Linear(256, action_num),
        )

    def is_reccurent(self):
        return False

    def q_function(self, state):
        h = F.relu(self.fc1(state))
        h = self.fc2(h)
        return h

