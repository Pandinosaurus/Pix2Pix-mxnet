from __future__ import print_function
import logging

class Trainer(object):
    logging.basicConfig(level=logging.INFO)

    def __init__(self, trainee, epochs=400, resume=True):
        assert trainee
        self.epochs = epochs
        self.passed_iterations = 0
        self.processed_data_entries = 0
        self.trainee = trainee
        self.checkpoint_every_epoch = 1
        self.visualize_every_epoch = 1
        self.resume = resume
        self.resume_epoch = 5

    def save_progress(self, epoch):
        self.trainee.save_progress(epoch)

    def resume_progress(self, epoch):
        self.trainee.resume_progress(epoch)

    def visualize_progress(self):
        self.trainee.visualize_progress()

    def do_train(self):
        if self.resume:
            self.resume_progress(self.resume_epoch)

        for epoch in range(0, self.epochs):
            self.trainee.run_iteration(epoch)

            if epoch % self.checkpoint_every_epoch == 0:
                self.save_progress(epoch)
            if epoch % self.visualize_every_epoch == 0:
                self.visualize_progress()
