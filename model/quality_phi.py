import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class QualityPhi(Chain):
    def __init__(self, input_dim, action_num):
        super(QualityPhi, self).__init__(
            fch=L.Linear(nobias=True, in_size=input_dim, out_size=input_dim),
            fcg=L.Linear(nobias=True, in_size=input_dim, out_size=action_num),
        )

    def __call__(self, h, o):
        left, right = F.split_axis(self.fch(h) + o, 2, axis=1)
        g = F.concat((F.relu(left), right), axis=1)
        #g = F.leaky_relu(self.fch(h) + o, slope=0.5)
        q = self.fcg(g)
        return q
