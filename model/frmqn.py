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

class FRMQN(Chain):
    def __init__(self, input_num, action_num, max_buff_size, m, e):
        assert(m == e)
        print("FRMQN Model", input_num, action_num)
        super(FRMQN, self).__init__(
            memory_module = MemoryModule(max_buff_size=max_buff_size, m=m, e=e),
            encoder=L.Linear(in_size=input_num, out_size=e),
            context=L.LSTM(in_size=(e+m), out_size=m),
            quality=QualityPhi(m, action_num),
        )
        self.o = None

    def q_function(self, state):
        e = self.encoder(state)
        self.memory_module.write(e)
        if self.o is None:
            self.o = self.xp.zeros(e.shape, np.float32)
        h = self.context(F.concat((e, self.o)))
        self.o, p = self.memory_module.read(h)
        q = self.quality(h, self.o)
        if state.shape[0] == 1:
            print("p", p.data)

        return q


    def is_reccurent(self):
        return True

    def has_memory_module(self):
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

