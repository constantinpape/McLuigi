import unittest
import luigi
import os

import z5py
import vigra

from mc_luigi import PipelineParameter
from mc_luigi.dataTasks import StackedRegionAdjacencyGraph
from test_class import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    # TODO without ignore label
    def test_rag(self):
        ppl_parameter = PipelineParameter()
        ppl_parameter.read_input_file('./inputs.json')
        ppl_parameter.useN5Backend = True
        ppl_parameter.ignoreSegLabel = 0
        inputs = ppl_parameter.inputs
        seg = inputs["seg"]
        luigi.run(["--local-scheduler",
                   "--pathToSeg", seg],
                  StackedRegionAdjacencyGraph)

        n_nodes_expected = z5py.File(seg, use_zarr_format=False)['data'][:].max() + 1
        rag_path = './cache/StackedRegionAdjacencyGraph_sampleA_watershed.h5'
        self.assertTrue(os.path.exists(rag_path))
        n_nodes = vigra.readHDF5(rag_path, 'numberOfNodes')
        self.assertEqual(n_nodes_expected, n_nodes)


if __name__ == '__main__':
    unittest.main()
