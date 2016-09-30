import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from collections import deque
import copy

class MemoryModule(Chain):
    def __init__(self, max_buff_size, m, e):
        super(MemoryModule, self).__init__(
            W_key=L.Linear(nobias=True, in_size=e, out_size=m),
            W_val=L.Linear(nobias=True, in_size=e, out_size=m),
        )
        self.max_buff_size = max_buff_size
        self.m = m
        self.e = e

        self.key_buff = deque()
        self.val_buff = deque()

    def reset_state(self):
        self.key_buff.clear()
        self.val_buff.clear()

    def write(self, enc):
        assert(len(self.key_buff) == len(self.val_buff))
        #print(enc.shape)
        self.key_buff.append(self.W_key(enc))
        self.val_buff.append(self.W_val(enc))
        if self.max_buff_size < len(self.key_buff):
            self.key_buff.popleft()
            self.val_buff.popleft()
        assert(len(self.key_buff) == len(self.val_buff))

    def read(self, h):
        #M_key = F.swapaxes(F.stack(self.key_buff, axis=0), axis1=0, axis2=1) # (B, M, m)
        M_key = F.stack(self.key_buff, axis=1) # (B, M, m)

        self.p = F.softmax(F.reshape(F.batch_matmul(M_key, h, transa=False, transb=False), (h.shape[0], M_key.shape[1]))) # (B, M)
        #p = F.reshape(p, (h.shape[0], 1, M_key.shape[1])) # (B, 1, M)
        #print("p", p.shape)
        #M_val = F.swapaxes(F.stack(self.val_buff, axis=0), axis1=0, axis2=1) # (B, M, m)
        M_val = F.stack(self.val_buff, axis=1) # (B, M, m)
        #print("M_val", M_val.shape)
        o = F.batch_matmul(self.p, M_val, transa=True, transb=False) # (B, 1, m)
        o = F.reshape(o, (o.shape[0], o.shape[2])) # (B, m)
        #print("o", o.shape)
        return o, self.p

    def push_state(self):
        self.key_buff_back = copy.copy(self.key_buff)
        self.val_buff_back = copy.copy(self.val_buff)
        self.reset_state()

    def pop_state(self):
        self.key_buff = self.key_buff_back
        self.val_buff = self.val_buff_back
