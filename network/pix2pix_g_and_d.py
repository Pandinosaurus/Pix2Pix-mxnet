import mxnet as mx
import math

def get_pix2pix_unet_generator(network_params):
    """
    This generator would produce 256x256 images and it uses
    U-net architecture
    https://arxiv.org/pdf/1505.04597.pdf
    :return:
    """
    e0 = mx.sym.Variable('gData')
    e1 = mx.sym.Convolution(e0,
                            kernel=network_params.kernel_size,
                            stride=network_params.stride_size,
                            pad=network_params.padding_size,
                            num_filter=network_params.ngf,
                            no_bias=True)
    e2 = mx.sym.LeakyReLU(e1, slope=0.2)
    e2 = mx.sym.Convolution(e2,
                            kernel=network_params.kernel_size,
                            stride=network_params.stride_size, pad=network_params.padding_size,
                            num_filter=network_params.ngf * 2, no_bias=True)
    e2 = mx.sym.BatchNorm(e2)
    e3 = mx.sym.LeakyReLU(e2, act_type='leaky', slope=0.2)

    # 3
    e3 = mx.sym.Convolution(e3,
                            kernel=network_params.kernel_size,
                            stride=network_params.stride_size, pad=network_params.padding_size,
                            num_filter=network_params.ngf * 4, no_bias=network_params.no_bias)
    e3 = mx.sym.BatchNorm(e3)
    e4 = mx.sym.LeakyReLU(e3, act_type="leaky", slope=0.2)
    network = mx.sym.Convolution(e4,
                                 kernel=network_params.kernel_size,
                                 stride=network_params.stride_size, pad=network_params.padding_size,
                                 num_filter=network_params.ngf * 8, no_bias=network_params.no_bias)
    e4 = mx.sym.BatchNorm(network)
    network = mx.sym.LeakyReLU(e4, act_type="leaky", slope=0.2)

    network = mx.sym.Convolution(network,
                                 kernel=network_params.kernel_size,
                                 stride=network_params.stride_size, pad=network_params.padding_size,
                                 num_filter=network_params.ngf * 8, no_bias=network_params.no_bias)
    e5 = mx.sym.BatchNorm(network)
    network = mx.sym.LeakyReLU(e5, act_type="leaky", slope=0.2)
    # 6
    network = mx.sym.Convolution(network,
                                 kernel=network_params.kernel_size,
                                 stride=network_params.stride_size, pad=network_params.padding_size,
                                 num_filter=network_params.ngf * 8, no_bias=network_params.no_bias)
    e6 = mx.sym.BatchNorm(network)
    network = mx.sym.LeakyReLU(e6, act_type="leaky", slope=0.2)

    # 7
    network = mx.sym.Convolution(network,
                                 kernel=network_params.kernel_size,
                                 stride=network_params.stride_size, pad=network_params.padding_size,
                                 num_filter=network_params.ngf * 8, no_bias=network_params.no_bias)
    e7 = mx.sym.BatchNorm(network)
    network = mx.sym.LeakyReLU(e7,
                               act_type="leaky", slope=network_params.slope
                               )

    # 8
    network = mx.sym.Convolution(network,
                                 kernel=network_params.kernel_size,
                                 stride=network_params.stride_size, pad=network_params.padding_size,
                                 num_filter=network_params.ngf * 8, no_bias=network_params.no_bias)
    e8 = mx.sym.BatchNorm(network)

    # Decoder
    # 1
    network = mx.sym.Activation(e8, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 8)
    network = mx.sym.BatchNorm(network)
    network = mx.sym.Dropout(network, p=0.5)
    decoder_one = network
    network = mx.sym.Concat(decoder_one, e7, dim=1)

    # 2
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 8)
    network = mx.sym.BatchNorm(network)
    network = mx.sym.Dropout(network, p=0.5)
    decoder_two = network
    network = mx.sym.Concat(decoder_two, e6, dim=1)

    # 3
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 8)
    network = mx.sym.BatchNorm(network)
    network = mx.sym.Dropout(network, p=0.5)
    decoder_three = network
    network = mx.sym.Concat(decoder_three, e5, dim=1)

    # 4
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 8)
    network = mx.sym.BatchNorm(network)
    decoder_four = network
    network = mx.sym.Concat(decoder_four, e4, dim=1)

    # 5
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 4)
    network = mx.sym.BatchNorm(network)
    decoder_five = network
    network = mx.sym.Concat(decoder_five, e3, dim=1)

    # 6
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf * 2)
    network = mx.sym.BatchNorm(network)
    decoder_six = network
    network = mx.sym.Concat(decoder_six, e2, dim=1)

    # 7
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size, num_filter=network_params.ngf)
    network = mx.sym.BatchNorm(network)
    decoder_seven = network
    network = mx.sym.Concat(decoder_seven, e1, dim=1)

    # 8
    network = mx.sym.Activation(network, act_type="relu")
    network = mx.sym.Deconvolution(network,
                                   kernel=network_params.kernel_size, stride=network_params.stride_size,
                                   pad=network_params.padding_size,
                                   num_filter=network_params.nc)  # nc = 2 for colorization

    network = mx.sym.Activation(network, act_type="tanh")
    return network


def get_pix2pix_discriminator(network_params, num_layers=3):
    # default filter size is enough to build receptive fields that would cover
    data = mx.sym.Variable('dData')
    label = mx.sym.Variable('label')

    network = data
    for index in range(0, num_layers):
        # number of dimensions we transform our picture. Initially we have 3 dimensions ( picture channels )

        if index == 0:  # first layer of the network
            # we do BatchNorm only on layers > 1
            network = mx.sym.Convolution(network,
                                         kernel=network_params.kernel_size,
                                         stride=network_params.stride_size,
                                         pad=network_params.padding_size,
                                         num_filter=network_params.ndf,
                                         no_bias=True)

            network = mx.sym.LeakyReLU(network,
                                       act_type="leaky", slope=0.2)
        else:
            num_filters_multiplier = int(min(math.pow(2, index), 8))
            network = mx.sym.Convolution(network,
                                         kernel=network_params.kernel_size,
                                         stride=network_params.stride_size, pad=network_params.padding_size,
                                         num_filter=network_params.ndf * num_filters_multiplier, no_bias=True)
            network = mx.sym.BatchNorm(network)
            network = mx.sym.LeakyReLU(network, act_type="leaky", slope=network_params.slope)

    num_filters_last = int(min(math.pow(2, num_layers), 8))

    network = mx.sym.Convolution(network,
                                 kernel=(4, 4), stride=(1, 1), pad=(1, 1),
                                 num_filter=network_params.ndf * num_filters_last, no_bias=True)

    network = mx.sym.BatchNorm(network)

    network = mx.sym.LeakyReLU(network, slope=0.2)

    network = mx.sym.Convolution(network,
                                 kernel=network_params.kernel_size,
                                 stride=(1, 1), pad=(1, 1),
                                 num_filter=1, no_bias=True)
    # LogisticRegressionOutput
    # defines cost as cost = -(y*log(P)+(1-y)*log(1-P))
    # in a form of ||sigmoid(pre_out)-y|| which is equivalent
    # sigmoid(pred_out) (1-sigmoid(pred_out))
    # please see https://github.com/apache/incubator-mxnet/issues/2001 for details
    dloss = mx.sym.LogisticRegressionOutput(data=network, label=label)

    return dloss