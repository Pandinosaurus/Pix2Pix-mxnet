import argparse

def parse_startup_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colorize_file_name", default=None, help="path to file to colorize")
    parser.add_argument("--input_dir", default="./train_data/city.rec", help="path to folder containing images")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--output_dir", default="./", help="where to put output files")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs", default=400)
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")

    parser.add_argument("--resume_training", type=bool, default=False,
                        help="Whether or not to continue training a model")
    parser.add_argument("--resume_position", type=int, default=500,
                        help="Epoch or iteration to pick from previous save to continue training")
    parser.add_argument("--checkpoint_freq", type=int, default=100,
                        help="Save a checkpoint every {checkpoint_freq} iteration")

    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--lambda1", type=float, default=100, help="Lambda for training of L1 loss")
    parser.add_argument("--l1_weight", type=float, default=100, help="weight on L1 term for generator gradient")

    parser.add_argument("--gpu_ctx", type=bool, default=False, help="Whether to use GPU or CPU")
    parser.add_argument("--colorize", type=bool, default=True, help="Whether to use LAB colorization in training")

    options = parser.parse_args()
    assert options
    assert options.lr is not None
    assert options.beta1 is not None
    assert options.max_epochs is not None
    return options