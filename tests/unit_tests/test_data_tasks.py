import unittest
from subprocess import call
from shutil import rmtree

import z5py

from test_class import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def check_wsresult(self, res_path):
        result = z5py.File(res_path, use_zarr_format=False)['data'][:]
        self.assertEqual(result.shape, self.expected_shape)
        prev_offset = -1
        for z, ws in enumerate(result):
            wsmin = ws.min()
            self.assertEqual(wsmin, prev_offset + 1)
            prev_offset = ws.max()
        rmtree(res_path)

    def check_wsresult_masked(self, res_path):
        result = z5py.File(res_path, use_zarr_format=False)['data'][:]
        self.assertEqual(result.shape, self.expected_shape)
        prev_offset = 0
        for z, ws in enumerate(result):
            print("!!!!", z, "!!!!!")
            wsmasked = ws[ws != 0]
            wsmin = wsmasked.min()
            self.assertEqual(wsmin, prev_offset + 1)
            prev_offset = wsmasked.max()
        rmtree(res_path)

    def _test_wsdt_default(self):
        call(['python', './executables/watershed.py', 'default'])
        res_path = './cache/WsdtSegmentation_sampleA_affinitiesXY.n5'
        self.check_wsresult(res_path)

    def _test_wsdt_nominseg(self):
        call(['python', './executables/watershed.py', 'nominseg'])
        res_path = './cache/WsdtSegmentation_sampleA_affinitiesXY.n5'
        self.check_wsresult(res_path)

    def test_wsdt_masked(self):
        call(['python', './executables/watershed.py', 'masked'])
        res_path = './cache/WsdtSegmentation_sampleA_affinitiesXY.n5'
        self.check_wsresult_masked(res_path)

    def _test_wsdt_masked_nominseg(self):
        call(['python', './executables/watershed.py', 'masked_nominseg'])
        res_path = './cache/WsdtSegmentation_sampleA_affinitiesXY.n5'
        self.check_wsresult_masked(res_path)


if __name__ == '__main__':
    unittest.main()
