import unittest
import os
import vigra
import numpy as np
from subprocess import call

from testClass import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()
        call(['python', 'task_wrapper.py', 'learn_rf'])

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def check_segmentation(self, result):
        self.assertEqual(result.shape, self.expected_shape)
        clusters = np.unique(result)
        self.assertGreater(len(clusters), self.expected_shape[0])

    def test_multicut_wf(self):
        call(['python', 'task_wrapper.py', 'mc'])
        seg_path = './cache/MulticutSegmentation_standard.h5'
        self.assertTrue(os.path.exists(seg_path))
        self.check_segmentation(vigra.readHDF5(seg_path, 'data'))

    def test_blockwise_wf(self):
        call(['python', 'task_wrapper.py', 'blockwise_mc'])
        seg_path = './cache/BlockwiseMulticutSegmentation_L1_20_256_256_5_50_50_standard.h5'
        self.assertTrue(os.path.exists(seg_path))
        self.check_segmentation(vigra.readHDF5(seg_path, 'data'))


if __name__ == '__main__':
    unittest.main()
