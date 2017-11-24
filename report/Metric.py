import ml_metrics
import mxnet as mx


class Metric:

    def __init__(self):
        self.accuracy = mx.metric.CustomMetric(Metric.accuracy)
        self.squared_error = mx.metric.CustomMetric(ml_metrics.se)

    def update_accuracy(self, label, output):
        self.accuracy.update([label, ], [output, ])

    def get_accuracy(self):
        return self.accuracy.get()

    def reset_accuracy(self):
        self.accuracy.reset()

    def update_se(self, label, output):
        self.squared_error.update([label, ], [output, ])

    def get_se(self):
        return self.squared_error.get()

    def reset_sey(self):
        self.squared_error.reset()

    @staticmethod
    def accuracy(label, prediction):
        pred = prediction.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()


