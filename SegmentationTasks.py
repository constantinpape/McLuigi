import luigi

from DataTasks import ExternalInputData

from wsdt import WsdtSegmentation

class WsdtSegmentation2D(luigi.Task):
    """
    Task for generating segmentation via wsdt.
    """

    PathToProbabilities = luigi.Parameter()
    WatershedParameter = luigi.DictParameter()

    def requires(self):
        """
        Dependencies:
        """
        return ExternalInputData(self.PathToProbabilities)

    def run(self):
        # TODO run wsdt
        pass

    def output(self):
        # TODO proper saving
        pass

class WsdtSegmentation3D(luigi.Task):
    """
    Task for generating segmentation via wsdt.
    """

    PathToProbabilities = luigi.Parameter()
    WatershedParameter = luigi.DictParameter()

    def requires(self):
        """
        Dependencies:
        """
        return ExternalInputData(self.PathToProbabilities)

    def run(self):
        # TODO run wsdt
        pass

    def output(self):
        # TODO proper saving
        pass
