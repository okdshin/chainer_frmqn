import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from .quality_phi import QualityPhi
from .memory_module import MemoryModule

class RMQN(Chain):
    def __init__(self, input_num, action_num, max_buff_size, m, e):
        print("RMQN Model", input_num, action_num)
        super(RMQN, self).__init__(
            memory_module = MemoryModule(max_buff_size=max_buff_size, m=m, e=e),
            encoder=L.Linear(in_size=input_num, out_size=e),
            context=L.LSTM(in_size=e, out_size=m),
            quality=QualityPhi(m, action_num),
        )

    def q_function(self, state):
        e = self.encoder(state)
        self.memory_module.write(e)
        h = self.context(e)
        o, p = self.memory_module.read(h)
        q = self.quality(h, o)
        if state.shape[0] == 1:
            print("p", p.data)

        return q


    def is_reccurent(self):
        return True

    def reset_state(self):
        self.memory_module.reset_state()
        self.context.reset_state()

    def push_state(self):
        self.memory_module.push_state()
        self.h_back = self.context.h
        self.c_back = self.context.c
        self.context.reset_state()

    def pop_state(self):
        self.memory_module.pop_state()
        self.context.h = self.h_back
        self.context.c = self.c_back

