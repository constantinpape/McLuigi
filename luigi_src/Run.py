import luigi
import os

import logging
import json

from PipelineParameter import PipelineParameter
from LearningTasks import EdgeProbabilitiesFromExternalRF


class SimpleTest(luigi.Task):

    PathInp = luigi.Parameter()
    PathSeg = luigi.Parameter()
    FeatureParameter = luigi.DictParameter()

    def requires(self):
        return EdgeFeatures(self.PathInp, self.PathSeg,
            self.FeatureParameter )

    def run(self):
        logging.info("Running SimpleTest")
        data = self.input().read()
        logging.info("shape:" + str(data.shape))

    def output(self):
        return luigi.LocalTarget("dummy.txt")


class TestEdgeProbs(luigi.Task):

    PathRF = luigi.Parameter()

    def requires(self):
        return EdgeProbabilitiesFromExternalRF( self.PathRF )

    def run(self):
        logging.info("Running EdgeProbsTest")
        data = self.input().read()
        logging.info("shape:" + str(data.shape))



if __name__ == '__main__':

    PipelineParameter().InputFile = "../config/input_config.json"
    PipelineParameter().FeatureConfigFile = "../config/feature_config.json"

    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)

    luigi.run(["--local-scheduler",
        "--PathRF", inputs["rf"]],
        main_task_cls = TestEdgeProbs)
