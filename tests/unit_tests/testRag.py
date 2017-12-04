import unittest
import luigi
import os

from mc_luigi import PipelineParameter
from mc_luigi.dataTasks import StackedRegionAdjacencyGraph
from testClass import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def test_rag(self):
        ppl_parameter = PipelineParameter()
        ppl_parameter.read_input_file('./inputs.json')
        ppl_parameter.useN5Backend = False
        inputs = ppl_parameter.inputs
        seg = inputs["seg"][0]

        # TODO get central scheduler running
        luigi.run(["--local-scheduler",
                   "--pathToSeg", seg],
                  StackedRegionAdjacencyGraph)
        self.assertTrue(os.path.exists('./cache/StackedRegionAdjacencyGraph_seg.h5'))


if __name__ == '__main__':
    unittest.main()
