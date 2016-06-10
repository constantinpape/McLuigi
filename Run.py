import luigi
import os

import logging

from FeatureTasks import EdgeFeatures


class SimpleTest(luigi.Task):

    PathInp = luigi.Parameter()
    PathSeg = luigi.Parameter()
    ParameterFolder = luigi.Parameter()

    def requires(self):
        return EdgeFeatures(self.PathInp, self.PathSeg,
                os.path.join(self.ParameterFolder, "feature_config.json") )

    def run(self):
        logging.info("Running SimpleTest")
        data = self.input().read()

    def output(self):
        return luigi.LocalTarget("dummy.txt")



if __name__ == '__main__':

    luigi.run(["--local-scheduler",
        "--PathInp", "/home/constantin/Work/home_hdd/data/test_block/test-raw.h5",
        "--PathSeg", "/home/constantin/Work/home_hdd/data/test_block/test-seg.h5",
        "--ParameterFolder", "/home/constantin/Work/luigi_cut/config"],
        main_task_cls = SimpleTest)
