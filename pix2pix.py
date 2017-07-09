import math
import cv2
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from zope.interface import implementer

from l1_loss import get_l1_absolute_loss
from neural_network_interface import NeuralNetworkInterface
from pix2pix_metrics import entropy, accuracy, error
from util.lab_color_utils_mx import rgb_to_lab, lab_to_rgb
from util.process_lab_utils_mx import preprocess_lab, deprocess_lab
from util.image_iter import ImageIter


@implementer(NeuralNetworkInterface)
class Pix2Pix(object):
    def __init__(self):
        # Great explanation of Convolutional network parameters
        # https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
        self.kernel_size = (4, 4)  # look at a square of 4x4 pixels to form a filter
        self.stride_size = (2, 2)  # ?
        self.padding_size = (1, 1)  # ?
        self.slope = 0.2  # ?
        self.no_bias = True  # ?
        self.fix_gamma = True  # ?
        self.ngf = 64  # starting number of generator filters
        self.ndf = 64  # starting number of discriminator filters
        self.batch_size = 1  # we train on one image at a time
        self.nc = 2  # number of channels in an image
        self.lr = 0.0002
        self.epochs = 400
        self.beta1 = 0.5
        self.ctx = mx.gpu(0)  # working on gpu 0
        self.check_point = True
        self.temp_grad_discriminator = None
        self.temp_grad_generator = None
        self.real_a = mx.nd.empty((1, 1, 256, 256), self.ctx)
        self.real_b = mx.nd.empty((1, 2, 256, 256), self.ctx)
        self.fake_b = mx.nd.empty((1, 2, 256, 256), self.ctx)
        self.resumed = False

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

    def __setup_metrics(self):
        self.metric_generator_entropy = mx.metric.CustomMetric(entropy)
        self.metric_discriminator_entropy = mx.metric.CustomMetric(entropy)
        self.metric_accuracy = mx.metric.CustomMetric(accuracy)

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

            if count % 10 == 0:
                print('epoch:', epoch, 'count: ', count, 'metric: ',
                      self.metric_accuracy.get(),
                      self.metric_generator_entropy.get(),
                      self.metric_discriminator_entropy.get())

                fake_rgb = self.lab_parts_to_rgb(self.fake_b.asnumpy(), a_image.asnumpy(), ctx=self.ctx)
                # TODO: move to visualize_progress function
                cv2.imshow("Real lightness", np.squeeze(a_image.asnumpy(), axis=0).transpose((1, 2, 0)))
                cv2.imshow("Fake colorization", cv2.cvtColor(fake_rgb.asnumpy(), cv2.COLOR_BGR2RGB))
                cv2.imshow("Real image", cv2.cvtColor(nd.cast(real_a, dtype='uint8').asnumpy(), cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

    def lab_parts_to_rgb(self, image, brightness, ctx=None):
        if ctx is None:
            raise ValueError("ctx can not be None")

        image = np.squeeze(image, axis=0)
        brightness = np.squeeze(brightness, axis=0)
        a_chan, b_chan = np.split(image, 2, 0)
        a_chan = nd.array(a_chan, ctx=ctx)
        b_chan = nd.array(b_chan, ctx=ctx)
        brightness = nd.array(brightness, ctx=self.ctx)
        lab = deprocess_lab(brightness, a_chan, b_chan)
        rgb = lab_to_rgb(nd.array(np.squeeze(lab.asnumpy(), axis=0), ctx=ctx), ctx=ctx)
        return nd.cast(rgb * 256, dtype='uint8')

    def _get_pix2pix_unet_generator(self):
        """
        This generator would produce 256x256 images and it uses
        U-net architecture
        https://arxiv.org/pdf/1505.04597.pdf
        :return:
        """
        e0 = mx.sym.Variable('gData')
        e1 = mx.sym.Convolution(e0,
                                kernel=self.kernel_size,
                                stride=self.stride_size,
                                pad=self.padding_size,
                                num_filter=self.ngf,
                                no_bias=True)
        e2 = mx.sym.LeakyReLU(e1, slope=0.2)
        e2 = mx.sym.Convolution(e2,
                                kernel=self.kernel_size,
                                stride=self.stride_size, pad=self.padding_size,
                                num_filter=self.ngf * 2, no_bias=True)
        e2 = mx.sym.BatchNorm(e2)
        e3 = mx.sym.LeakyReLU(e2, act_type='leaky', slope=0.2)

        # 3
        e3 = mx.sym.Convolution(e3,
                                kernel=self.kernel_size,
                                stride=self.stride_size, pad=self.padding_size,
                                num_filter=self.ngf * 4, no_bias=self.no_bias)
        e3 = mx.sym.BatchNorm(e3)
        e4 = mx.sym.LeakyReLU(e3, act_type="leaky", slope=0.2)
        network = mx.sym.Convolution(e4,
                                     kernel=self.kernel_size,
                                     stride=self.stride_size, pad=self.padding_size,
                                     num_filter=self.ngf * 8, no_bias=self.no_bias)
        e4 = mx.sym.BatchNorm(network)
        network = mx.sym.LeakyReLU(e4, act_type="leaky", slope=0.2)

        network = mx.sym.Convolution(network,
                                     kernel=self.kernel_size,
                                     stride=self.stride_size, pad=self.padding_size,
                                     num_filter=self.ngf * 8, no_bias=self.no_bias)
        e5 = mx.sym.BatchNorm(network)
        network = mx.sym.LeakyReLU(e5, act_type="leaky", slope=0.2)
        # 6
        network = mx.sym.Convolution(network,
                                     kernel=self.kernel_size,
                                     stride=self.stride_size, pad=self.padding_size,
                                     num_filter=self.ngf * 8, no_bias=self.no_bias)
        e6 = mx.sym.BatchNorm(network)
        network = mx.sym.LeakyReLU(e6, act_type="leaky", slope=0.2)

        # 7
        network = mx.sym.Convolution(network,
                                     kernel=self.kernel_size,
                                     stride=self.stride_size, pad=self.padding_size,
                                     num_filter=self.ngf * 8, no_bias=self.no_bias)
        e7 = mx.sym.BatchNorm(network)
        network = mx.sym.LeakyReLU(e7,
                                   act_type="leaky", slope=self.slope
                                   )

        # 8
        network = mx.sym.Convolution(network,
                                     kernel=self.kernel_size,
                                     stride=self.stride_size, pad=self.padding_size,
                                     num_filter=self.ngf * 8, no_bias=self.no_bias)
        e8 = mx.sym.BatchNorm(network)

        # Decoder
        # 1
        network = mx.sym.Activation(e8, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 8)
        network = mx.sym.BatchNorm(network)
        network = mx.sym.Dropout(network, p=0.5)
        decoder_one = network
        network = mx.sym.Concat(decoder_one, e7, dim=1)

        # 2
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 8)
        network = mx.sym.BatchNorm(network)
        network = mx.sym.Dropout(network, p=0.5)
        decoder_two = network
        network = mx.sym.Concat(decoder_two, e6, dim=1)

        # 3
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 8)
        network = mx.sym.BatchNorm(network)
        network = mx.sym.Dropout(network, p=0.5)
        decoder_three = network
        network = mx.sym.Concat(decoder_three, e5, dim=1)

        # 4
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 8)
        network = mx.sym.BatchNorm(network)
        decoder_four = network
        network = mx.sym.Concat(decoder_four, e4, dim=1)

        # 5
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 4)
        network = mx.sym.BatchNorm(network)
        decoder_five = network
        network = mx.sym.Concat(decoder_five, e3, dim=1)

        # 6
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf * 2)
        network = mx.sym.BatchNorm(network)
        decoder_six = network
        network = mx.sym.Concat(decoder_six, e2, dim=1)

        # 7
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size, num_filter=self.ngf)
        network = mx.sym.BatchNorm(network)
        decoder_seven = network
        network = mx.sym.Concat(decoder_seven, e1, dim=1)

        # 8
        network = mx.sym.Activation(network, act_type="relu")
        network = mx.sym.Deconvolution(network,
                                       kernel=self.kernel_size, stride=self.stride_size,
                                       pad=self.padding_size,
                                       num_filter=self.nc)  # nc = 2 for colorization

        network = mx.sym.Activation(network, act_type="tanh")
        return network

    def _get_pix2pix_discriminator(self, num_layers=3):
        # default filter size is enough to build receptive fields that would cover
        data = mx.sym.Variable('dData')
        label = mx.sym.Variable('label')

        network = data
        for index in range(0, num_layers):
            # number of dimensions we transform our picture. Initially we have 3 dimensions ( picture channels )

            if index == 0:  # first layer of the network
                # we do BatchNorm only on layers > 1
                network = mx.sym.Convolution(network,
                                             kernel=self.kernel_size,
                                             stride=self.stride_size,
                                             pad=self.padding_size,
                                             num_filter=self.ndf,
                                             no_bias=True)

                network = mx.sym.LeakyReLU(network,
                                           act_type="leaky", slope=0.2)
            else:
                num_filters_multiplier = int(min(math.pow(2, index), 8))
                network = mx.sym.Convolution(network,
                                             kernel=self.kernel_size,
                                             stride=self.stride_size, pad=self.padding_size,
                                             num_filter=self.ndf * num_filters_multiplier, no_bias=True)
                network = mx.sym.BatchNorm(network)
                network = mx.sym.LeakyReLU(network, act_type="leaky", slope=self.slope)

        num_filters_last = int(min(math.pow(2, num_layers), 8))

        network = mx.sym.Convolution(network,
                                     kernel=(4, 4), stride=(1, 1), pad=(1, 1),
                                     num_filter=self.ndf * num_filters_last, no_bias=True)

        network = mx.sym.BatchNorm(network)

        network = mx.sym.LeakyReLU(network, slope=0.2)

        network = mx.sym.Convolution(network,
                                     kernel=self.kernel_size,
                                     stride=(1, 1), pad=(1, 1),
                                     num_filter=1, no_bias=True)
        # LogisticRegressionOutput
        # defines cost as cost = -(y*log(P)+(1-y)*log(1-P))
        # in a form of ||sigmoid(pre_out)-y|| which is equivalent
        # sigmoid(pred_out) (1-sigmoid(pred_out))
        # please see https://github.com/apache/incubator-mxnet/issues/2001 for details
        dloss = mx.sym.LogisticRegressionOutput(data=network, label=label)

        return dloss

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
        # self.real_b = target
        # D(x,y)
        self.label[:] = 1
        self.mod_discriminator.forward(mx.io.DataBatch([self.real_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()
        self.temp_grad_discriminator = [[grad.copyto(grad.context) for grad in grads] for grads in
                                        self.mod_discriminator._exec_group.grad_arrays]

        self.mod_discriminator.update_metric(self.metric_discriminator_entropy, [self.label])
        self.mod_discriminator.update_metric(self.metric_accuracy, [self.label])

        # Update discriminator on fake
        self.label[:] = 0
        self.mod_discriminator.forward(mx.io.DataBatch([self.fake_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()
        self.__sum_temp_grad_discriminator()

        self.mod_discriminator.update_metric(self.metric_discriminator_entropy, [self.label])
        self.mod_discriminator.update_metric(self.metric_accuracy, [self.label])

    def __forward_generator(self):
        # G* = arg min max Loss of cGAN (G,D) + train_lambda * Loss L1(G)
        #           G   D
        # L1 loss = self.mod_loss => absolute difference between real image and a generated fake
        # duh

        self.label[:] = 1
        self.mod_discriminator.forward(mx.io.DataBatch([self.fake_b], [self.label]), is_train=True)
        self.mod_discriminator.backward()  # back-propogate on the real error - adjust weights

        df_dg = mx.nd.split(self.mod_discriminator.get_input_grads()[0], axis=1, num_outputs=2)[1]

        self.fake_b.copyto(self.mod_loss.arg_dict['rec'])
        self.real_b.copyto(self.mod_loss.arg_dict['origin'])

        self.mod_loss.forward()
        self.mod_loss.backward()

        train_lambda = 100
        mod_loss_gradient_generator = mx.nd.multiply(self.mod_loss.grad_dict['rec'], train_lambda)

        combine = mx.nd.add(df_dg, mod_loss_gradient_generator)

        self.mod_generator.backward([combine])
        self.__save_temp_grad_generator()

        self.metric_generator_entropy.update([self.label], self.mod_discriminator.get_outputs())

    def setup(self):
        symbol_generator = self._get_pix2pix_unet_generator()
        symbol_discriminator = self._get_pix2pix_discriminator()
        self.train_iter = ImageIter(self.batch_size, (3, 256, 256), 'dData')
        self.label = mx.nd.zeros((1, 1, 30, 30), ctx=self.ctx)

        # L1 Absolute Loss module
        mod_loss = get_l1_absolute_loss()
        mod_loss = mod_loss.simple_bind(self.ctx, grad_req="write", origin=(1, 2, 256, 256))
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

        self.__setup_metrics()

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
