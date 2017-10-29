import math

import cv2
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from network.pix2pix_g_and_d import get_pix2pix_unet_generator, get_pix2pix_discriminator
from .neural_network_interface import NeuralNetworkInterface
from zope.interface import implementer

from network.l1_loss import get_l1_absolute_loss
from util.lab_color_utils_mx import rgb_to_lab
from util.process_lab_utils_mx import preprocess_lab
from util.process_lab_utils_np import lab_parts_to_rgb


plt.ion()
fig, ax = plt.subplots()


@implementer(NeuralNetworkInterface)
class Pix2Pix(object):
    def __init__(self, options):
        # Great explanation of Convolutional network parameters
        # https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
        self.kernel_size = (4, 4)  # look at a square of 4x4 pixels to form a filter
        self.stride_size = (2, 2)  # amount by which the filter shifts
        self.padding_size = (1, 1)  # ?
        self.slope = 0.2  # ?
        self.no_bias = True  # ?
        self.fix_gamma = True  # ?
        self.ngf = options.ngf if options.ngf else 64  # starting number of generator filters
        self.ndf = options.ngf if options.ndf else 64  # starting number of discriminator filters
        self.batch_size = options.batch_size if options.batch_size else 1  # we train on one image at a time
        self.nc = 2  # number of channels in an image
        self.lr = options.lr if options.lr else 0.0002
        self.epochs = options.max_epochs if options.max_epochs else 400
        self.beta1 = options.beta1 if options.beta1 else 400
        self.ctx = mx.gpu(0)  # working on gpu 0
        self.check_point = True
        self.temp_grad_discriminator = None
        self.temp_grad_generator = None
        self.real_a = mx.nd.empty((1, 1, 256, 256), self.ctx)
        self.real_b = mx.nd.empty((1, 2, 256, 256), self.ctx)
        self.fake_b = mx.nd.empty((1, 2, 256, 256), self.ctx)
        self.resumed = False
        self.opts = options
        self.l1_loss = 0

        assert self.opts
        assert self.opts.input_dir

    def save_progress(self, epoch):
        self.mod_discriminator.save_checkpoint("D", epoch, save_optimizer_states=True)
        self.mod_generator.save_checkpoint("G", epoch, save_optimizer_states=True)

    def resume_progress(self, epoch):
        self.mod_discriminator = mx.module.Module.load("D", epoch, load_optimizer_states=True, data_names=('dData',),
                                                       label_names=('label',), context=self.ctx)
        self.mod_discriminator.bind(data_shapes=[("dData", (self.batch_size,) + (2, 256, 256))],
                                    label_shapes=[('label', (self.batch_size, 1, 30, 30))],
                                    inputs_need_grad=True)
        self.mod_discriminator.init_optimizer(optimizer='adam',
                                              optimizer_params={
                                                  'learning_rate': self.lr,
                                                  'wd': 0.,
                                                  'beta1': 0.5,
                                              })

        self.mod_generator = mx.module.Module.load("G", epoch, load_optimizer_states=True, data_names=('gData',),
                                                   label_names=None, context=self.ctx)
        self.mod_generator.bind(data_shapes=[('gData', (self.batch_size,) + (1, 256, 256))])
        self.mod_generator.init_optimizer(optimizer='adam',
                                          optimizer_params={
                                              'learning_rate': self.lr,
                                              'wd': 0.,
                                              'beta1': 0.5,
                                          })

        self.resumed = True

        self.setup()

    def visualize_progress(self):
        pass

    def run_iteration(self, epoch):
        self._do_train_iteration(epoch)

    def _do_train_iteration(self, epoch):

        self.train_iter.reset()
        for count, batch in enumerate(self.train_iter):
            real_a = batch.data[0]
            real_a = real_a.transpose((0, 2, 3, 1))
            real_a = nd.array(np.squeeze(real_a.asnumpy(), axis=0), ctx=self.ctx)

            lab = rgb_to_lab(real_a, ctx=self.ctx)
            lightness_chan, a_chan, b_chan = preprocess_lab(lab)

            a_image = nd.expand_dims(lightness_chan, axis=2)
            a_image = a_image.transpose((3, 2, 0, 1))
            b_image = nd.stack(a_chan, b_chan, axis=2)
            b_image = nd.transpose(b_image, axes=(3, 2, 0, 1))

            inputs = a_image
            targets = b_image

            self.__create_real_fake(inputs, targets)
            self.__forward_backward()

            if count % 2000 == 0:
                ax.scatter(count, self.l1_loss)
                fig.canvas.flush_events()

                fake_rgb = lab_parts_to_rgb(self.fake_b.asnumpy(), a_image.asnumpy())
                # TODO: move to visualize_progress function
                cv2.imshow("Fake colorization", cv2.cvtColor(fake_rgb, cv2.COLOR_BGR2RGB))
                # cv2.imshow("Real image", cv2.cvtColor(nd.cast(real_a, dtype='uint8').asnumpy(), cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

        plt.ioff()
        plt.show()

    def __create_real_fake(self, inputs, targets):
        self.real_a = inputs
        self.real_b = targets
        self.mod_generator.forward(mx.io.DataBatch([self.real_a]), is_train=True)
        self.fake_b = self.mod_generator.get_outputs()[0].copy()

    def __forward_backward(self):
        self.__forward_discriminator()
        self.__forward_generator()
        self.mod_discriminator.update()
        self.mod_generator.update()

    def __forward_discriminator(self):
        # Update discriminator on real
        # here we update the weights of the discriminator network by showing it what is a real image and supplying real label to it
        # if it detects a real image as fake, it would update weights accordingly
        # D(x,y)

        # We train D to maximize the probability of assigning the correct label to both training examples and samples from G
        self.label[:] = 1
        self.mod_discriminator.forward(mx.io.DataBatch([self.real_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()

        # we get total discriminator gradient after backward o
        self.temp_grad_discriminator = [[grad.copyto(grad.context) for grad in grads] for grads in
                                        self.mod_discriminator._exec_group.grad_arrays]

        # Update discriminator on fake
        self.label[:] = 0
        self.mod_discriminator.forward(mx.io.DataBatch([self.fake_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()
        self.__sum_temp_grad_discriminator()


    def __forward_generator(self):
        # G* = arg min max Loss of cGAN (G,D) + train_lambda * Loss L1(G)
        #           G   D
        # L1 loss = self.mod_loss => absolute difference between real image and a generated fake
        # duh
        # We send fake image inpersonating as a real one to the discriminator,
        # and getting the error that we get from D(G(x))
        # the error we get from the discriminator shows how well discriminator can tell if something
        # is real or fake.

        # We simultaneously train G to minimize log(1 − D(G(z)))
        self.label[:] = 1
        self.mod_discriminator.forward(mx.io.DataBatch([self.fake_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()  # back-propogate on the real error - adjust weights

        df_dg = self.mod_discriminator.get_input_grads()[0]

        self.mod_loss.forward(mx.io.DataBatch([self.fake_b, self.real_b]), is_train=True)
        self.mod_loss.backward()

        self.l1_loss = self.mod_loss.get_outputs()[0].asnumpy().sum()
        train_lambda = 100
        df_doae = mx.nd.multiply(self.mod_loss.get_input_grads()[0], train_lambda)

        combine = df_dg + df_doae

        self.mod_generator.backward([combine])
        self.__save_temp_grad_generator()

    def setup(self):
        symbol_generator = get_pix2pix_unet_generator(self)
        symbol_discriminator = get_pix2pix_discriminator(self)
        self.train_iter = mx.image.ImageIter(
            1,
            (3, 256, 256),
            path_imgrec=self.opts.input_dir
        )

        self.label = mx.nd.zeros((1, 1, 30, 30), ctx=self.ctx)

        mod_loss = mx.mod.Module(symbol=get_l1_absolute_loss(), data_names=('origin','rec',), label_names=None,
                                         context=self.ctx)
        mod_loss.bind(data_shapes=[('origin', (self.batch_size,) + (2, 256, 256)), ('rec', (self.batch_size,) + (2, 256, 256))], inputs_need_grad=True)
        mod_loss.init_params(initializer=mx.init.Zero())
        mod_loss.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': self.lr,
                'wd': 0.,
                'beta1': 0.5,
            })

        self.mod_loss = mod_loss

        if not self.resumed:
            # =============module G=============
            module_generator = mx.mod.Module(symbol=symbol_generator, data_names=('gData',), label_names=None,
                                             context=self.ctx)
            module_generator.bind(data_shapes=[('gData', (self.batch_size,) + (1, 256, 256))])
            module_generator.init_params(initializer=mx.init.Normal(0.02))
            module_generator.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': self.lr,
                    'wd': 0.,
                    'beta1': 0.5,
                })

            self.mod_generator = module_generator
            # =============module D=============
            mod_discriminator = mx.mod.Module(symbol=symbol_discriminator, data_names=('dData',),
                                              label_names=('label',),
                                              context=self.ctx)
            mod_discriminator.bind(data_shapes=[("dData", (self.batch_size,) + (2, 256, 256))],
                                   label_shapes=[('label', (self.batch_size, 1, 30, 30))],
                                   inputs_need_grad=True)
            mod_discriminator.init_params(initializer=mx.init.Normal(0.02))
            mod_discriminator.init_optimizer(
                optimizer='adam',
                optimizer_params={
                    'learning_rate': self.lr,
                    'wd': 0.,
                    'beta1': 0.5,
                })

            self.mod_discriminator = mod_discriminator

        assert self.mod_loss
        assert self.mod_generator
        assert self.mod_discriminator
        assert self.train_iter

    def __save_temp_grad_generator(self):
        self.temp_grad_generator = [
            [grad.copyto(grad.context) for grad in grads]
            for grads in self.mod_generator._exec_group.grad_arrays]

    def __sum_temp_grad_discriminator(self):
        for gradsr, gradsf in zip(self.mod_discriminator._exec_group.grad_arrays, self.temp_grad_discriminator):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
