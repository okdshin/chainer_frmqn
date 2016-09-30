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
            cnn1=L.Convolution2D(in_channels=3*12, out_channels=32, ksize=4, stride=2, pad=1),
            cnn2=L.Convolution2D(in_channels=32, out_channels=64, ksize=4, stride=2, pad=1),
            fc1=L.Linear(4096, 256),
            fc2=L.Linear(256, action_num),
        )

    def is_reccurent(self):
        return False

    def q_function(self, state):
        state = F.reshape(state, (state.shape[0], 36, 32, 32))
        h = F.relu(self.cnn1(state))
        h = F.relu(self.cnn2(h))
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h

