import unittest
import vigra
import luigi

from mc_luigi import WsdtSegmentation, PipelineParameter
from testClass import McLuigiTestCase


class TestDataTasks(McLuigiTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataTasks, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        super(TestDataTasks, cls).tearDownClass()

    def check_wsresult(self, result):
        self.assertEqual(result.shape, self.expected_shape)
        prev_offset = -1
        for z, ws in enumerate(result):
            wsmin = ws.min()
            self.assertEqual(wsmin, prev_offset + 1)
            prev_offset = ws.max()

    def test_wsdt(self):
        ppl_parameter = PipelineParameter()
        ppl_parameter.read_input_file('./inputs.json')
        inp = ppl_parameter.inputs['data'][1]

        # TODO get central scheduler running
        luigi.run([
            "--local-scheduler",
            "--pathToProbabilities", inp],
            WsdtSegmentation
        )

        result = vigra.readHDF5('./cache/WsdtSegmentation_pmap.h5', 'data')
        self.check_wsresult(result)


if __name__ == '__main__':
    unittest.main()
