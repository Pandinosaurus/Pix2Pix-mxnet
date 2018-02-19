#### Pix2pix for image colorization
###### Python 3.6 and MXNet 1.0

This is an attempt to implement Pix2pix network using Python and MXNet
Mainly used to colorize black and white images, specifically videos.

The network is configured right now to work with input of 1x3x256x256 images.
An image will be split into lightness and a,b channels, the lightness channel
will serve as an input to generator which will attempt to produce a 1x2x256x256 a,b channels tensor
that we will feed into the discriminator in a sequence with original a,b channels.

#### Metrics visualization
tensorboard --logdir=./logs/train

##### Code is based on
https://github.com/affinelayer/pix2pix-tensorflow
https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/pixel2pixel.ipynb

##### pix2pix paper
https://arxiv.org/abs/1611.07004

####
Video colorization example:
![Video colorization, captured and colorized in real-time using notebook and usb web-cam](https://s3-us-west-1.amazonaws.com/pix2pix/real_time_video_colorization.gif)