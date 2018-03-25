import zope.interface


class NeuralNetworkInterface(zope.interface.Interface):
    """An interface for a neural network implementations. Defines common attributes and methods"""

    check_point = zope.interface.Attribute(
        """Indicates whether or not the network should start from the latest checkpoint""")

    batch_size = zope.interface.Attribute(
        """Define a batch size for the training""")

    def save_progress(self):
        """ Save the training progress """

    def resume_progress(self):
        """ Resume training from the last checkpoint """

    def run_iteration(self):
        """ Run a training iteration """
