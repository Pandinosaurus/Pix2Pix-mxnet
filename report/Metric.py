from report.AccuracyMetric import AccuracyMetric
from report.GradientMetric import GradientMetric
from report.SpeedMetric import SpeedMetric


class Metric:

    def __init__(self):
        self.speed = SpeedMetric()
        self.gradient = GradientMetric()
        self.accuracy = AccuracyMetric()

    def update_accuracy(self, label, output):
        self.accuracy.update(label, output)

    def log_accuracy(self):
        self.accuracy.log_accuracy()

    def log_speed(self, value):
        self.speed.log_speed(value)

    def log_gradient(self, network_name, value):
        self.gradient.log_gradient(network_name, value)

    def get_accuracy(self):
        return self.accuracy.get()

    def reset_accuracy(self):
        self.accuracy.reset()





