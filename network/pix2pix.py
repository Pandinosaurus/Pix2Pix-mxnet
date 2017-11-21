import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import time
import logging


from network.gluon_pix2pix_modules import UnetGenerator, Discriminator
from network.metric import facc
from util.process_lab_utils_np import lab_parts_to_rgb
from util.visual_utils import visualize
from .neural_network_interface import NeuralNetworkInterface
from zope.interface import implementer


from util.lab_color_utils_mx import rgb_to_lab
from util.process_lab_utils_mx import preprocess_lab


@implementer(NeuralNetworkInterface)
class Pix2Pix(object):

    plt.ion()

    def __init__(self, options):

        assert options

        logging.basicConfig(level=logging.DEBUG)

        self.options = options
        self.batch_size = options.batch_size

        self.ctx = mx.cpu(0) if not options.gpu_ctx else mx.gpu(0)

        self.train_iter = mx.image.ImageIter(
            1,
            (3, 256, 256),
            path_imgrec=options.input_dir
        )

        self.lr = options.lr if options.lr else 0.0002
        self.beta1 = options.beta1 if options.beta1 else 0.5
        self.lambda1 = options.lambda1 if options.lambda1 else 100

        # Losses
        self.GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
        self.L1_loss = gluon.loss.L1Loss()
        self.netG = None
        self.netD = None
        self.trainerG = None
        self.trainerD = None

        self.metric = mx.metric.CustomMetric(facc)

        self.stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    def setup(self):
        self.netG, self.netD, self.trainerG, self.trainerD = self.__set_network()
        assert self.netG
        assert self.netD
        assert self.trainerG
        assert self.trainerD

    def __param_init(self, param):
            if param.name.find('conv') != -1:
                if param.name.find('weight') != -1:
                    param.initialize(init=mx.init.Normal(0.02), ctx=self.ctx)
                else:
                    param.initialize(init=mx.init.Zero(), ctx=self.ctx)
            elif param.name.find('batchnorm') != -1:
                param.initialize(init=mx.init.Zero(), ctx=self.ctx)
                if param.name.find('gamma') != -1:
                    param.set_data(nd.random_normal(1, 0.02, param.data().shape))

    def __network_init(self, net):
            for param in net.collect_params().values():
                self.__param_init(param)

    def __set_network(self):

            if self.options.colorize:
                final_out = 2
            else:
                final_out = 3

            net_g = UnetGenerator(in_channels=1, num_downs=8, final_out=final_out)
            net_d = Discriminator(in_channels=3)

            self.__network_init(net_g)
            self.__network_init(net_d)

            trainer_g = gluon.Trainer(net_g.collect_params(), 'adam', {'learning_rate': self.lr, 'beta1': self.beta1})
            trainer_d = gluon.Trainer(net_d.collect_params(), 'adam', {'learning_rate': self.lr, 'beta1': self.beta1})

            return net_g, net_d, trainer_g, trainer_d

    def save_progress(self, epoch):
        filename_net_d = "netD{0}".format(epoch)
        filename_net_g = "netG{0}".format(epoch)
        self.netD.save_params(filename_net_d)
        self.netG.save_params(filename_net_g)

    def resume_progress(self, epoch):
        pass

    def visualize_progress(self):
        pass

    def run_iteration(self, epoch):

        if self.options.colorize:
            self.__do_train_iteration_colorization(epoch)
        else:
            pass

    def __do_train_iteration_colorization(self, epoch):

        epoch_tic = time.time()
        batch_tic = time.time()

        self.train_iter.reset()
        for count, batch in enumerate(self.train_iter):

            real_in, real_out = self.__prepare_real_in_real_out(batch=batch)
            fake_out, fake_concat = self.__get_fake_out_fake_concat(real_in)
            self.__maximize_discriminator(fake_concat, real_in, real_out)
            self.trainerD.step(batch.data[0].shape[0])

            self.__minimize_generator(real_in, real_out, fake_out)
            self.trainerG.step(batch.data[0].shape[0])

            if count % 10 == 0:

                visualize(lab_parts_to_rgb(fake_out, real_in, ctx=self.ctx))

                name, acc = self.metric.get()
                logging.info('speed: {} samples/s'.format(self.batch_size / (time.time() - batch_tic)))
                logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                    % (nd.mean(self.err_d).asscalar(),
                       nd.mean(self.err_g).asscalar(), acc, count, epoch))

                batch_tic = time.time()

        name, acc = self.metric.get()
        self.metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - epoch_tic))

    def __prepare_real_in_real_out(self, batch):
        real_a = batch.data[0]
        real_a = real_a.transpose((0, 2, 3, 1))
        real_a = nd.array(np.squeeze(real_a.asnumpy(), axis=0), ctx=self.ctx)
        lab = rgb_to_lab(real_a, ctx=self.ctx)
        lightness_chan, a_chan, b_chan = preprocess_lab(lab)

        real_in = nd.expand_dims(lightness_chan, axis=2)
        real_in = real_in.transpose((3, 2, 0, 1))

        real_out = nd.stack(a_chan, b_chan, axis=2)
        real_out = nd.transpose(real_out, axes=(3, 2, 0, 1))

        return real_in, real_out

    def __get_fake_out_fake_concat(self, real_in):
        fake_out = self.netG(real_in)
        fake_concat = nd.concat(real_in, fake_out, dim=1)
        return fake_out, fake_concat

    def __maximize_discriminator(self, fake_concat, real_in, real_out):
        ############################
        # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
        ###########################
        with autograd.record():
            # Train with fake image
            output = self.netD(fake_concat)
            fake_label = nd.zeros(output.shape, ctx=self.ctx)
            err_d_fake = self.GAN_loss(output, fake_label)
            self.metric.update([fake_label, ], [output, ])

            # Train with real image
            real_concat = nd.concat(real_in, real_out, dim=1)
            output = self.netD(real_concat)
            real_label = nd.ones(output.shape, ctx=self.ctx)
            err_d_real = self.GAN_loss(output, real_label)
            err_d = (err_d_real + err_d_fake) * 0.5
            err_d.backward()
            self.err_d = err_d
            self.metric.update([real_label, ], [output, ])

    def __minimize_generator(self, real_in, real_out, fake_out):
        ############################
        # (2) Update G network: minimize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
        ###########################
        with autograd.record():
            fake_out = self.netG(real_in)
            fake_concat = nd.concat(real_in, fake_out, dim=1)
            output = self.netD(fake_concat)
            real_label = nd.ones(output.shape, ctx=self.ctx)
            err_g = self.GAN_loss(output, real_label) + self.L1_loss(real_out, fake_out) * self.lambda1
            err_g.backward()
            self.err_g = err_g








