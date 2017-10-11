from __future__ import print_function
import mxnet as mx

class ImageIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, data_var, opts):
        self.internal = mx.io.ImageRecordIter(
            path_imgrec=opts.input_dir,
            data_shape=data_shape,
            batch_size=batch_size,
            rand_crop   = True,
            rand_mirror = True,
            max_crop_size = 256,
            min_crop_size = 256)
        self.provide_data = [(data_var, (batch_size,) + data_shape)]
        self.provide_label = []
        self.ctx = mx.gpu(0)

    def reset(self):
        self.internal.reset()

    def iter_next(self):
        return self.internal.iter_next()

    def getdata(self):
        return [self.internal.getdata()]