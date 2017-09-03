import unittest
import luigi
import json
import os

from mc_luigi import LearnClassifierFromGt, PipelineParameter
from testClass import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def test_learnrf(self):
        ppl_parameter = PipelineParameter()
        ppl_parameter.read_input_file('./inputs.json')
        inputs = ppl_parameter.inputs

        # TODO get central scheduler running
        luigi.run([
            "--local-scheduler",
            "--pathsToSeg", json.dumps(inputs["seg"]),
            "--pathsToGt", json.dumps(inputs["gt"])],
            LearnClassifierFromGt
        )
        self.assertTrue(os.path.exists(inputs["rf"]))


if __name__ == '__main__':
    unittest.main()
