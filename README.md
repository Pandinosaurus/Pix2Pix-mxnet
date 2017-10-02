### Pix2pix for image colorization, Python 3.6 and MXNet 0.11
This is an attempt to implement Pix2pix network using Python and MXNet
Mainly used to colorize black and white images, specifically videos.

The network is configured right now to work with input of 1x3x256x256 images.
An image will be split into lightness and a,b channels, the lightness channel
will serve as an input to generator which will attempt to produce a 1x2x256x256 a,b channels tensor
that we will feed into the discriminator in a sequence with original a,b channels.


### code borrows from
https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/Pix2Pix
https://github.com/tqchen/mxnet-gan
https://github.com/affinelayer/pix2pix-tensorflow

### pix2pix
https://arxiv.org/abs/1611.07004