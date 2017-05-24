from regression_tests_utils import regression_test

import os
import subprocess
import vigra
import json


#
# Reference values from LCC branch of nm-pipeline
#

# vi-split, vi-merge, adapted rand
reference_values_mc = {
    'sampleA_0': (0.6, 0.5, 0.11),  # 0.4018, 0.3106, 0.0951
    'sampleA_1': (0.5, 0.5, 0.12),  # 0.3884, 0.3404, 0.1040
    'sampleB_0': (1.1, 1.1, 0.33),  # 0.9968, 0.9271, 0.3148
    'sampleB_1': (0.8, 0.8, 0.15),  # 0.6517, 0.6571, 0.1329
    'sampleC_0': (1.2, 0.6, 0.25),  # 1.0260, 0.4707, 0.2318
    'sampleC_1': (1.2, 0.6, 0.175),  # 0.9935, 0.4526, 0.1582
}

# not used here for now
reference_values_lmc = {
    'sampleA_0': (0.6, 0.4, 0.11),  # 0.3953, 0.3188, 0.0963
    'sampleA_1': (0.5, 0.5, 0.12),  # 0.3835, 0.3404, 0.0994
    'sampleB_0': (1.2, 0.9, 0.25),  # 1.004, 0.7219, 0.2277
    'sampleB_1': (0.8, 0.6, 0.08),  # 0.6362, 0.4594, 0.0688
    'sampleC_0': (1.1, 0.5, 0.20),  # 0.9582, 0.3543, 0.1855
    'sampleC_1': (1.1, 0.5, 0.16),  # 0.9661, 0.3917, 0.1454
}


def regression_test_cremi(samples):

    # run all multicuts
    for ds_test in samples:

        train_inputs = './cremi_inputs/%s/train_files.json' % ds_test
        assert os.path.exists(train_inputs), train_inputs

        test_inputs = './cremi_inputs/%s/test_files.json' % ds_test
        assert os.path.exists(test_inputs), test_inputs

        subprocess.call([
            'python',
            'learn.py',
            train_inputs
        ])

        subprocess.call([
            'python',
            'mc.py',
            test_inputs
        ])

    print "Eval Cremi"
    for ds_test in samples:

        vi_split_ref, vi_merge_ref, adapted_ri_ref = reference_values_mc[ds_test]

        test_inputs = './cremi_inputs/%s/test_files.json' % ds_test
        assert os.path.exists(test_inputs), test_inputs

        with open(test_inputs) as f:
            in_files = json.load(f)
            gt_p = in_files['gt']
            mc_p = os.path.join(in_files['cache'], 'MulticutSegmentation.h5')
        gt = vigra.readHDF5(gt_p, 'data')
        mc_seg = vigra.readHDF5(mc_p, 'data')

        print "Regression Test MC for %s..." % ds_test
        regression_test(
            gt,
            mc_seg,
            vi_split_ref,
            vi_merge_ref,
            adapted_ri_ref
        )


if __name__ == '__main__':
    samples = ['sample%s_%i' % (sample, i) for sample in ('A', 'B', 'C') for i in (0, 1)]
    regression_test_cremi(samples)
