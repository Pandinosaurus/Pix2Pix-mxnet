from __future__ import print_function
import logging

class Trainer(object):
    logging.basicConfig(level=logging.INFO)

    def __init__(self, trainee, epochs=400, resume=True, options=None):
        assert trainee
        assert options
        self.opts = options
        self.trainee = trainee
        self.epochs = options.max_epochs
        self.checkpoint_every_epoch = options.checkpoint_freq
        self.resume = options.resume_training
        self.resume_epoch = 1

    def save_progress(self, epoch):
        self.trainee.save_progress(epoch)

    def resume_progress(self, epoch):
        self.trainee.resume_progress(epoch)

    def setup_network(self):
        self.trainee.setup()

    def do_train(self):

        print(self.opts)

        if self.resume:
            self.resume_progress(self.resume_epoch)
        else:
            self.setup_network()

        for epoch in range(0, self.epochs):
            self.trainee.run_iteration(epoch)

            if epoch % self.checkpoint_every_epoch == 0:
                self.save_progress(epoch)
