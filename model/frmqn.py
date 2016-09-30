import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from functools import reduce

from .quality_phi import QualityPhi
from .memory_module import MemoryModule

class FRMQN(Chain):
    def __init__(self, input_num, action_num, max_buff_size, m, e):
        print("FRMQN Model", input_num, action_num)
        super(FRMQN, self).__init__(
            cnn1=L.Convolution2D(in_channels=3, out_channels=32, ksize=4, stride=2, pad=1),
            cnn2=L.Convolution2D(in_channels=32, out_channels=64, ksize=4, stride=2, pad=1),
            memory_module = MemoryModule(max_buff_size=max_buff_size, m=m, e=e),
            context=L.LSTM(in_size=(e+m), out_size=m),
            quality=QualityPhi(m, action_num),
        )
        self.m = m
        self.o = None

    def q_function(self, state):
        batch_size = state.shape[0]
        e = F.relu(self.cnn1(state))
        e = F.relu(self.cnn2(e))
        self.memory_module.write(e)
        if self.o is None:
            self.o = self.xp.zeros((batch_size, self.m), np.float32)
        e = F.reshape(e, (batch_size, reduce(lambda x,y: x*y, e.shape[1:])))
        h = self.context(F.concat((e, self.o)))
        self.o, p = self.memory_module.read(h)
        q = self.quality(h, self.o)
        if state.shape[0] == 1:
            print("p", p.data)

        return q


    def is_reccurent(self):
        return True

    def reset_state(self):
        self.memory_module.reset_state()
        self.context.reset_state()
        self.o = None

    def push_state(self):
        self.memory_module.push_state()
        self.h_back = self.context.h
        self.c_back = self.context.c
        self.context.reset_state()
        self.o_back = self.o
        self.o = None

    def pop_state(self):
        self.memory_module.pop_state()
        self.context.h = self.h_back
        self.context.c = self.c_back
        self.o = self.o_back

