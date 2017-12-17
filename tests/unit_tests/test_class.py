import unittest
import json
import os


class McLuigiTestCase(unittest.TestCase):

    # shape of cremi sample A
    expected_shape = (125, 1447, 1353)
    folder_on_fs = '/home/consti/Work/data_neuro/cremi_n5'

    # TODO make upload for cremi sample A n5 data
    @staticmethod
    def donwload_test_data():
        import zipfile
        from subprocess import call
        url = 'https://www.dropbox.com/s/l1tgzlim8h1pb7w/test_data_anisotropic.zip?dl=0'
        data_file = 'data.zip'

        # FIXME good old wget still does the job done better than any python lib I know....
        call(['wget', '-O', data_file, url])
        with zipfile.ZipFile(data_file) as f:
            f.extractall('.')
        os.remove(data_file)

    @classmethod
    def setUpClass(cls):
        super(McLuigiTestCase, cls).setUpClass()

        if os.path.exists(cls.folder_on_fs):
            path = cls.folder_on_fs
        else:
            cls.donwload_test_data()
            path = './data'
        input_dict = {'data': [os.path.join(path, 'sampleA_raw.n5'),
                               os.path.join(path, 'sampleA_affinitiesXY.n5'),
                               os.path.join(path, 'sampleA_affinitiesZ.n5')],
                      'seg': os.path.join(path, 'sampleA_watershed.n5'),
                      'gt': os.path.join(path, 'sampleA_gt.n5'),
                      'mask': os.path.join(path, 'sampleA_mask.n5'),
                      'rf': './cache/LearnClassifierFromGt_SingleInput',
                      'cache': './cache'}
        with open('./inputs.json', 'w') as f:
            json.dump(input_dict, f)

    @classmethod
    def tearDownClass(cls):
        super(McLuigiTestCase, cls).tearDownClass()
        from shutil import rmtree
        if os.path.exists('./data'):
            rmtree('./data')
        rmtree('./cache')
        os.remove('./inputs.json')
        os.remove('./luigi_workflow.log')
