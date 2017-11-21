import os
import tarfile
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import utils


class DataSetUtil:

    def __init__(self, options, img_wd=256, img_ht=256):
        assert options
        self.dataset = options.dataset

        self.img_wd = img_wd
        self.img_ht = img_ht

        self.train_img_path = '../%s/train' % self.dataset
        self.val_img_path = '../%s/val' % self.dataset

    def download_data(self):
        if not os.path.exists(self.dataset):
            url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz' % self.dataset
            os.mkdir(self.dataset)
            data_file = utils.download(url)
            with tarfile.open(data_file) as tar:
                tar.extractall(path='.')
            os.remove(data_file)

    def load_data(self, path, batch_size, is_reversed=False):
        img_in_list = []
        img_out_list = []
        for path, _, fnames in os.walk(path):
            for fname in fnames:
                if not fname.endswith('.jpg'):
                    continue
                img = os.path.join(path, fname)
                img_arr = mx.image.imread(img).astype(np.float32) / 127.5 - 1
                img_arr = mx.image.imresize(img_arr, self.img_wd * 2, self.img_ht)
                # Crop input and output images
                img_arr_in, img_arr_out = [mx.image.fixed_crop(img_arr, 0, 0, self.img_wd, self.img_ht),
                                           mx.image.fixed_crop(img_arr, self.img_wd, 0, self.img_wd, self.img_ht)]
                img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2, 0, 1)),
                                           nd.transpose(img_arr_out, (2, 0, 1))]
                img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                           img_arr_out.reshape((1,) + img_arr_out.shape)]
                img_in_list.append(img_arr_out if is_reversed else img_arr_in)
                img_out_list.append(img_arr_in if is_reversed else img_arr_out)

        return mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0), nd.concat(*img_out_list, dim=0)],
                                 batch_size=batch_size)
