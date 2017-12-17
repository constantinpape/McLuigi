import unittest
import os
import numpy as np
from subprocess import call

import z5py

from testClass import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()
        call(['python', './executables/workflow.py', 'learn_rf'])

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def check_segmentation(self, res_path):
        self.assertTrue(os.path.exists(res_path))
        result = z5py.File(res_path, use_zarr_format=False)['data'][:]
        self.assertEqual(result.shape, self.expected_shape)
        clusters = np.unique(result)
        self.assertGreater(len(clusters), self.expected_shape[0])

    def test_multicut_wf(self):
        call(['python', './executables/workflow.py', 'mc'])
        seg_path = './cache/MulticutSegmentation_standard.h5'
        self.check_segmentation(seg_path)

    def test_blockwise_wf(self):
        call(['python', './executables/workflow.py', 'blockwise_mc'])
        seg_path = './cache/BlockwiseMulticutSegmentation_L1_20_256_256_5_50_50_standard.h5'
        self.check_segmentation(seg_path)


if __name__ == '__main__':
    unittest.main()
