from trainer import Trainer
from pix2pix import Pix2Pix

if __name__ == '__main__':
    pix2pix = Pix2Pix()
    trainer = Trainer(trainee=pix2pix)
    trainer.do_train()