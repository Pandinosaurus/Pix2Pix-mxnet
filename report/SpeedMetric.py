from tensorboard.writer import SummaryWriter


class SpeedMetric:
    def __init__(self):
        self.training_log = 'logs/train/speed'
        self.speed_metric_name = 'batch_training_in_sec'
        self.summary_writer = SummaryWriter(self.training_log)

    def log_speed(self, speed_scalar):
        assert self.summary_writer
        self.summary_writer.add_scalar(self.speed_metric_name, speed_scalar)

