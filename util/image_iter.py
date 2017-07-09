from __future__ import print_function
import mxnet as mx

class ImageIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, dataVar):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec='./train_data/prefix_train.rec',
            data_shape=data_shape,
            batch_size=batch_size,
            rand_crop   = True,
            rand_mirror = True,
            max_crop_size = 256,
            min_crop_size = 256)
        self.provide_data = [(dataVar, (batch_size,) + data_shape)]
        self.provide_label = []
        self.ctx = mx.gpu(0)

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        return [self.internal.getdata()]