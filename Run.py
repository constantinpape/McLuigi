import luigi
import os

import logging

from FeatureTasks import EdgeFeatures
from LearningTasks import EdgeProbabilities


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

    PathInp = luigi.Parameter()
    PathSeg = luigi.Parameter()
    FeatureTasks = luigi.ListParameter()

    def requires(self):
        return EdgeProbabilities(self.PathInp, self.PathSeg,
            self.FeatureParameter )

    def run(self):
        logging.info("Running EdgeProbsTest")
        data = self.input().read()
        logging.info("shape:" + str(data.shape))



if __name__ == '__main__':
    with open("./config/feature_config.json",'r') as f:
        json = f.read().replace('\n','')

    print type(json)

    raw_path = "/home/consti/Work/data_neuro/test_block/test-raw.h5"
    seg_path = "/home/consti/Work/data_neuro/test_block/test-seg.h5"

    FeatureTasks = [EdgeFeatures(input_path, seg_path, json)]

    luigi.run(["--local-scheduler",
        "--PathInp", raw_path,
        "--PathSeg", seg_path,
        "--FeatureTasks", FeatureTasks],
        main_task_cls = TestEdgeProbs)

