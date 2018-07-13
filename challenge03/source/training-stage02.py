#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:33:34 2018

@author: quanle
"""

from IPython import display
from PIL import Image
from chainer.cuda import to_cpu
from chainer.dataset import concat_examples
from chainer.functions import mean_squared_error
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.serializers import save_npz
from glob import glob
from time import time
# from utils01 import Dataset, Model
import numpy as np

from PIL import Image, ImageOps
from chainer import ChainList
from chainer.dataset import DatasetMixin
import chainer.functions as F
import chainer.links as L
import numpy as np

class Dataset(DatasetMixin):
    def __init__(self, fp):
        super(Dataset, self).__init__()

        self.fp = fp

    def __len__(self):
        return len(self.fp[0])

    def get_example(self, i):
        #x = Image.open(self.fp[0][i]).convert('RGB').resize((138, 200), Image.LANCZOS)
        x = Image.open(self.fp[0][i]).convert('L').resize((138, 200), Image.LANCZOS)
        y = Image.open(self.fp[1][i]).convert('L').resize((138, 200), Image.LANCZOS)
# 
#         return np.asarray(x, 'f').transpose(2, 0, 1), np.asarray(y, 'f')[None]
        return np.asarray(x, 'f')[None], np.asarray(y, 'f')[None]

class ResidualBlock(ChainList):
    def __init__(self):
        super(ResidualBlock, self).__init__(
            L.Convolution2D(128, 128, 3, pad = 1),
            L.BatchNormalization(128),
            L.Convolution2D(128, 128, 3, pad = 1),
            L.BatchNormalization(128)
        )

    def __call__(self, x):
        return x + self[3](self[2](F.relu(self[1](self[0](x)))))

    
class ResidualBlock_64(ChainList):
    def __init__(self):
        super(ResidualBlock_64, self).__init__(
            L.Convolution2D(64, 64, 3, pad = 1),
            L.BatchNormalization(64),
            L.Convolution2D(64, 64, 3, pad = 1),
            L.BatchNormalization(64)
        )

    def __call__(self, x):
        return x + self[3](self[2](F.relu(self[1](self[0](x)))))
    
    
class Model(ChainList):
    def __init__(self):
        super(Model, self).__init__(
            L.Convolution2D(1, 32, 9, pad = 4, nobias = True),
            L.BatchNormalization(32),
            L.Convolution2D(32, 64, 3, 2, 1, True),
            L.BatchNormalization(64),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            L.Deconvolution2D(64, 32, 3, 2, 1, True, (200, 138)),
            L.BatchNormalization(32),
            L.Convolution2D(32, 1, 9, pad = 4)
        )

    def __call__(self, x):
        for i in range(len(self)):
#             print x.shape
            x = F.relu(self[i](x)) if i in (1, 3, 5, 12, 14) else self[i](x)

        return 127.5 * F.tanh(x) + 127.5
    
batch_size = 4
device = 0 # -1 for CPU or GPU ID (0, 1, etc.) for GPU
fp = (sorted(glob('../data_set/training_output/*.jpg')), sorted(glob('../data_set/training_ground-truth/*.jpg')))
frequency = 100 # display the results and save the model every [frequency] iterations
iterations = 1000000

dataset = Dataset(fp)
iterator = SerialIterator(dataset, batch_size)

model = Model() if device < 0 else Model().to_gpu(device)
optimizer = Adam()

optimizer.setup(model)
tic = time()

for i, batch in enumerate(iterator):
    x, y = concat_examples(batch, device)
    y_hat = model(x)
#     print (y.shape ,y_hat.shape)
    loss = mean_squared_error(y, y_hat)
    loss_org = mean_squared_error(y, x)

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if (i + 1) % frequency == 0:
        display.clear_output()
        print('iteration: {}, loss: {}, loss_org: {},  time: {}.'.format(i + 1, float(loss.data), float(loss_org.data), time() - tic))
        save_npz('../model_step02.npz', model.copy().to_cpu())

    if i + 1 == iterations:
        break
