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

class MQN(Chain):
    def __init__(self, input_num, action_num, max_buff_size, m, e):
        print("MQN Model", input_num, action_num)
        super(MQN, self).__init__(
            memory_module = MemoryModule(max_buff_size=max_buff_size, m=m, e=e),
            encoder=L.Linear(in_size=input_num, out_size=e),
            context=L.Linear(nobias=True, in_size=e, out_size=m),
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
    def push_state(self):
        self.memory_module.push_state()

    def pop_state(self):
        self.memory_module.pop_state()

