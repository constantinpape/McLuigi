import unittest
import json
import os


class McLuigiTestCase(unittest.TestCase):

    expected_shape = (29, 512, 512)

    @classmethod
    def setUpClass(cls):
        super(McLuigiTestCase, cls).setUpClass()
        import zipfile
        from subprocess import call
        url = 'https://www.dropbox.com/s/l1tgzlim8h1pb7w/test_data_anisotropic.zip?dl=0'
        data_file = 'data.zip'

        # FIXME good old wget still does the job done better than any python lib I know....
        call(['wget', '-O', data_file, url])
        with zipfile.ZipFile(data_file) as f:
            f.extractall('.')
        os.remove(data_file)

        input_dict = {
            'data': ['./data/raw.h5', './data/pmap.h5'],
            'seg': ['./data/seg.h5'],
            'gt': ['./data/gt.h5'],
            'rf': './cache/LearnClassifierFromGt_SingleInput',
            'cache': './cache'
        }
        with open('./inputs.json', 'w') as f:
            json.dump(input_dict, f)

    @classmethod
    def tearDownClass(cls):
        super(McLuigiTestCase, cls).tearDownClass()
        from shutil import rmtree
        rmtree('./data')
        rmtree('./cache')
        os.remove('./inputs.json')
        os.remove('./luigi_workflow.log')
