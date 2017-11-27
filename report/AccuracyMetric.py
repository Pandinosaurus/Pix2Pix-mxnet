from collections import namedtuple
import mxnet as mx


class AccuracyMetric:
    def __init__(self):
        self.accuracy_training_log = 'logs/train/accuracy'
        self.batch_end_logger = mx.contrib.tensorboard.LogMetricsCallback(self.accuracy_training_log)
        self.accuracy = mx.metric.CustomMetric(self.accuracy)
        self.accuracy_log = namedtuple('Accuracy', ['eval_metric'])
        self.accuracy_log.eval_metric = self.accuracy

    def log_accuracy(self):
            self.batch_end_logger.__call__(self.accuracy_log)

    def reset(self):
        self.accuracy.reset()

    def get(self):
        return self.accuracy.get()

    def update(self, label, output):
        self.accuracy.update([label, ], [output, ])

    def accuracy(self, label, prediction):
        pred = prediction.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()


