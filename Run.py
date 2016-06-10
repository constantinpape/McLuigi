import luigi
import os

import logging

from FeatureTasks import EdgeFeatures


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



if __name__ == '__main__':
    with open("./config/feature_config.json",'r') as f:
    #with open("./conf_simple.json",'r') as f:
        json = f.read()#.replace('\n','')

    print type(json)

    luigi.run(["--local-scheduler",
        "--PathInp", "/home/constantin/Work/home_hdd/data/test_block/test-raw.h5",
        "--PathSeg", "/home/constantin/Work/home_hdd/data/test_block/test-seg.h5",
        "--FeatureParameter", json],
        main_task_cls = SimpleTest)

