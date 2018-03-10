import os
from network.pix2pix import Pix2Pix
from options import parse_startup_arguments
from trainer import Trainer


os.environ['MXNET_CPU_WORKER_NTHREADS'] = "32"

if __name__ == '__main__':
    options = parse_startup_arguments()
    pix2pix = Pix2Pix(options)
    trainer = Trainer(trainee=pix2pix, options=options)
    trainer.do_train()
