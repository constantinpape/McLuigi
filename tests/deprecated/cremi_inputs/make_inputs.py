import json
import os
from itertools import chain

datasets = ['%s_%i' % (sample, i) for i in (0, 1)
            for sample in ('sampleA', 'sampleB', 'sampleC')]


def make_inputs(ds_test):
    top_folder = '/home/constantin/Work/neurodata_hdd/regression_test_data/cremi/'
    if not os.path.exists(ds_test):
        os.mkdir(ds_test)

    # make train inputs
    raw = [os.path.join(os.path.join(top_folder, ds), '%s_raw_train.h5' % ds) for ds in datasets if ds != ds_test]
    pmap = [os.path.join(os.path.join(top_folder, ds), '%s_pmap_train.h5' % ds) for ds in datasets if ds != ds_test]
    train_dict = {
        'data': list(chain.from_iterable(zip(raw, pmap))),
        'seg': [os.path.join(os.path.join(top_folder, ds), '%s_seg_train.h5' % ds)
                for ds in datasets if ds != ds_test],
        'gt': [os.path.join(os.path.join(top_folder, ds), '%s_gt_train.h5' % ds)
               for ds in datasets if ds != ds_test],
        'cache': '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/%s_train2' % ds_test
    }
    for key, paths in train_dict.iteritems():
        if key == 'cache':
            continue
        for p in paths:
            assert os.path.exists(p), p

    train_file = os.path.join(ds_test, 'train_files.json')
    with open(train_file, 'w') as f:
        json.dump(train_dict, f)

    test_dict = {
        'data': [os.path.join(top_folder, ds_test) + '/%s_raw_train.h5' % ds_test,
                 os.path.join(top_folder, ds_test) + '/%s_pmap_train.h5' % ds_test],
        'seg': os.path.join(os.path.join(top_folder, ds_test), '%s_seg_train.h5' % ds_test),
        'rf': '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/%s_train2/LearnClassifierFromGt_MultipleInput' % ds_test,
        'gt': os.path.join(os.path.join(top_folder, ds_test), '%s_gt_train.h5' % ds_test),
        'cache': '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/%s_test2' % ds_test
    }
    for key, paths in test_dict.iteritems():
        if key in ('cache', 'rf'):
            continue
        if key == 'data':
            for p in paths:
                assert os.path.exists(p), p
        else:
            assert os.path.exists(paths), paths

    test_file = os.path.join(ds_test, 'test_files.json')
    with open(test_file, 'w') as f:
        json.dump(test_dict, f)


if __name__ == '__main__':
    for ds in datasets:
        print ds
        make_inputs(ds)
