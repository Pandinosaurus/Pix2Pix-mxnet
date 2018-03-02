
import time
import logging
from datetime import datetime

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
import numpy as np


from network.gluon_pix2pix_modules import UnetGenerator, Discriminator
from report.Metric import Metric
from util.process_lab_utils_np import lab_parts_to_rgb
from util.visual_utils import visualize_cv2
from .neural_network_interface import NeuralNetworkInterface
from zope.interface import implementer
from util.lab_color_utils_mx import rgb_to_lab
from util.process_lab_utils_mx import preprocess_lab


@implementer(NeuralNetworkInterface)
class Pix2Pix(object):

    def __init__(self, options):

        assert options

        logging.basicConfig(level=logging.DEBUG)

        self.options = options

        self.metrics = Metric()

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
        self.err_g = None
        self.err_d = None

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
            net.collect_params().setattr('grad_req', 'write')
            for param in net.collect_params().values():
                self.__param_init(param)

    def __set_network(self):

            if self.options.colorize:
                final_out = 2
                in_channels = 1
            else:
                final_out = 3
                in_channels = 3

            net_g = UnetGenerator(in_channels=3, num_downs=8, final_out=final_out)
            net_d = Discriminator(in_channels=6, use_sigmoid=False)

            self.__network_init(net_g)
            self.__network_init(net_d)

            trainer_g = gluon.Trainer(net_g.collect_params(), 'adam', {'learning_rate': self.lr, 'beta1': self.beta1})
            trainer_d = gluon.Trainer(net_d.collect_params(), 'adam', {'learning_rate': self.lr, 'beta1': self.beta1})

            return net_g, net_d, trainer_g, trainer_d

    def save_progress(self, position):
        filename_net_d = "netD{0}".format(position)
        filename_net_g = "netG{0}".format(position)
        self.netD.save_params(filename_net_d)
        self.netG.save_params(filename_net_g)

    def resume_progress(self, position):
        filename_net_d = "netD{0}".format(position)
        filename_net_g = "netG{0}".format(position)
        self.netD.load_params(filename_net_d, ctx=self.ctx)
        self.netG.load_params(filename_net_g, ctx=self.ctx)

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

            self.trainerD.step(batch.data[0].shape[0], )
            self.__minimize_generator(real_in, real_out, fake_out)

            self.trainerG.step(batch.data[0].shape[0])

            training_speed_sec = time.time() - batch_tic


            visualize_cv2("Fake", lab_parts_to_rgb(fake_out, real_in, ctx=self.ctx))

            name, acc = self.metrics.get_accuracy()

            self.metrics.log_accuracy()
            self.metrics.log_speed(training_speed_sec)

            logging.info('samples per second: {}'.format(self.batch_size / (time.time() - batch_tic)))

            logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                    % (nd.mean(self.err_d).asscalar(),
                       nd.mean(self.err_g).asscalar(), acc, count, epoch))

            if count % self.options.checkpoint_freq == 0:
                self.save_progress(count)

            batch_tic = time.time()

        name, acc = self.metrics.get_accuracy()

        self.metrics.reset_accuracy()

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
            self.metrics.update_accuracy(fake_label, output)

            # Train with real image
            real_concat = nd.concat(real_in, real_out, dim=1)
            output = self.netD(real_concat)
            real_label = nd.ones(output.shape, ctx=self.ctx)
            err_d_real = self.GAN_loss(output, real_label)
            self.err_d = (err_d_real + err_d_fake) * 0.5
            self.err_d.backward()

            self.metrics.update_accuracy(real_label, output)

            first_layer_gradient = self.__get_incoming_gradient(self.netD)
            self.metrics.log_gradient('discriminator', first_layer_gradient)

    def __minimize_generator(self, real_in, real_out, fake_out):
        ############################
        # (2) Update G network: minimize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
        ###########################
        with autograd.record():

            # print(self.netG.collect_params())
            fake_out = self.netG(real_in)
            fake_concat = nd.concat(real_in, fake_out, dim=1)
            output = self.netD(fake_concat)
            real_label = nd.ones(output.shape, ctx=self.ctx)
            self.err_g = self.GAN_loss(output, real_label) + self.L1_loss(real_out, fake_out) * self.lambda1
            self.err_g.backward()
            first_layer_gradient = self.__get_incoming_gradient(self.netG)
            self.metrics.log_gradient('generator', first_layer_gradient)

    def __get_incoming_gradient(self, network):
        return list(network.collect_params().values())[0].list_grad()[0].asnumpy().flatten()








