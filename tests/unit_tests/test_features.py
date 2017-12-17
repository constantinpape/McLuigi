import unittest
import os
from subprocess import call

import z5py
import vigra

from test_class import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def check_features(self, feature_path):
        rag_path = './cache/StackedRegionAdjacencyGraph_sampleA_watershed.h5'
        self.assertTrue(os.path.exists(rag_path))
        n_edges = vigra.readHDF5(rag_path, 'numberOfEdges')

        self.assertTrue(os.path.exists(feature_path))
        features = z5py.File(feature_path, use_zarr_format=False)['data'][:]
        self.assertEqual(n_edges, len(features))
        for feat_id in range(features.shape[1]):
            self.assertFalse((features[:, feat_id] == 0).all())

    def test_region_features(self):
        call(['python', './executables/features.py', 'region'])
        feat_path = ''
        self.check_features(feat_path)


if __name__ == '__main__':
    unittest.main()
