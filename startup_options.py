import argparse


def parse_startup_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--colorize_file_name", default=None, help="path to file to colorize")
    parser.add_argument("--client_checkpoint_generative_model_prefix", default="G", help="prefix of the checkpoint file of generative model ")
    parser.add_argument("--client_checkpoint_generative_model_epoch", type=int, default=5, help="epoch of the checkpoint file of generative model ")
    parser.add_argument("--visualize_colorization", type=bool, default=True,
                        help="Whether to display client colorization or not")
    parser.add_argument("--save_colorization", type=bool, default=True,
                        help="Whether to save colorization result or not")
    parser.add_argument("--client_colorization_folder",  default="./",
                        help="Folder with client colorization data")

    parser.add_argument("--input_dir", default="./", help="path to folder containing images")
    parser.add_argument("--mode", default="train", choices=["train", "test", "validate"])
    parser.add_argument("--output_dir", default="./", help="where to put output files")
    parser.add_argument("--checkpoint_folder", default=None,
                        help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs", default=400)
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="Save a checkpoint every {freq} epoch")
    parser.add_argument("--visualize_freq", type=int, default=50, help="display progress every {freq} steps")
    parser.add_argument("--save_image_output_freq", type=int, default=1,
                        help="write current training images every {freq} epochs")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--lab_colorization", action="store_true",
                        help="split input image into brightness {inputs} and a,b color channels {targets}")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--training_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286,
                        help="scale images to this size before cropping to 256x256")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--resume_training", type=bool, default=True, help="Whether or not to continue training a model")

    # export options
    parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

    options = parser.parse_args()

    assert options.lr is not None
    assert options.beta1 is not None
    assert options.output_filetype is not None


    return options