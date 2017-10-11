import cv2
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from cv2_utils import show_mxnet_to_numpy_array
from startup_options import parse_startup_arguments
from util import rgb_to_lab
from util.process_lab_utils_mx import preprocess_lab
from util.process_lab_utils_np import lab_parts_to_rgb

ctx = mx.gpu(0) if mx.gpu(0) else mx.cpu()

def load_generator_from_checkpoint(options):
    file_prefix = options.client_checkpoint_generative_model_prefix
    training_epoch = 0

    mod_generator = mx.module.Module.load(file_prefix, training_epoch, load_optimizer_states=True, data_names=('gData',),
                                               label_names=None, context=ctx)
    mod_generator.bind(data_shapes=[('gData', (options.batch_size,) + (1, 256, 256))])
    mod_generator.init_optimizer(optimizer='adam',
                                      optimizer_params={
                                          'learning_rate': options.lr,
                                          'wd': 0.,
                                          'beta1': options.beta1,
                                      })

    return mod_generator

def load_image(image_filename):
    assert image_filename
    with open(image_filename, 'rb') as fp:
        str_image = fp.read()

    return mx.img.imdecode(str_image)

def get_color_channels(image):

    w, h = image.shape[0:2]

    if image.shape[0:1] != (256, 256):
        image = mx.image.resize_short(image, 256)
        image = mx.image.center_crop(image, (256, 256))[0]

    lightness_chan = get_lightness(image)

    grayscale_lightness = nd.expand_dims(lightness_chan, axis=2).transpose((3, 2, 0, 1))
    model.forward(mx.io.DataBatch([grayscale_lightness]), is_train=False)
    output = model.get_outputs()[0].copy()
    output = nd.array(np.squeeze(output.asnumpy(), axis=0))
    output = output.transpose((1,2,0))
    output = cv2.resize(output.asnumpy(), (w, h), interpolation=cv2.INTER_CUBIC)
    output = nd.expand_dims(nd.array(output), axis=2).transpose((2, 3, 1, 0))
    return output

def get_lightness(image):
    image = image.as_in_context(ctx)
    lab = rgb_to_lab(image, ctx=ctx)
    lightness_chan, _, _ = preprocess_lab(lab)
    return lightness_chan


def colorize_image(image_filename, model):
    real_a = load_image(image_filename)

    colorchannels = get_color_channels(real_a)
    grayscale_lightness = nd.expand_dims(get_lightness(real_a), axis=2).transpose((2,3,0,1))

    print(colorchannels.shape)
    print(grayscale_lightness.shape)

    fake_rgb = lab_parts_to_rgb(colorchannels.asnumpy(), grayscale_lightness.asnumpy())

    return fake_rgb

if __name__ == '__main__':
        options = parse_startup_arguments()
        file_to_colorize = options.colorize_file_name

        if file_to_colorize is None:
            file_to_colorize = "tupac.jpg"

        model = load_generator_from_checkpoint(options)
        image =  colorize_image(file_to_colorize, model)

        if options.visualize_colorization:
            show_mxnet_to_numpy_array("Fake colorization", image)


