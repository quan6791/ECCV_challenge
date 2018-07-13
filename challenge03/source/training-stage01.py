#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:07:44 2018

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
from utils import Dataset, Model
import numpy as np

batch_size = 4
device = 0 # -1 for CPU or GPU ID (0, 1, etc.) for GPU
fp = (sorted(glob('../data_set/training_input/*.jpg')), sorted(glob('../data_set/training_ground-truth/*.jpg')))
frequency = 100 # display the results and save the model every [frequency] iterations
iterations = 2000000

dataset = Dataset(fp)
iterator = SerialIterator(dataset, batch_size)

model = Model() if device < 0 else Model().to_gpu(device)
optimizer = Adam()

optimizer.setup(model)

tic = time()

for i, batch in enumerate(iterator):
    x, y = concat_examples(batch, device)
    y_hat = model(x)
    loss = mean_squared_error(y, y_hat)

    model.cleargrads()
    loss.backward()
    optimizer.update()

    if (i + 1) % frequency == 0:
        display.clear_output()
        print('iteration: {}, loss: {}, time: {}.'.format(i + 1, float(loss.data), time() - tic))
        save_npz('../model.npz', model.copy().to_cpu())

    if i + 1 == iterations:
        break