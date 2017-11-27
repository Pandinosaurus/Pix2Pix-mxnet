from tensorboard import summary
from tensorboard import FileWriter


class GradientMetric:
    def __init__(self):
        self.training_log = 'logs/train/network'
        self.speed_metric_name = '{0}_gradient'
        self.summary_writer = FileWriter(self.training_log)

    def log_gradient(self, network_name, gradient):
        assert self.summary_writer
        summary_value = summary.histogram('{0}'.format(network_name), gradient)
        self.summary_writer.add_summary(summary_value)

