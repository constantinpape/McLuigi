from volumina_viewer import volumina_n_layer
import vigra
import json
import os
import numpy as np


def check_projection(sample):

    in_file = './cremi_inputs/%s/test_files.json' % sample
    with open(in_file) as f:
        inputs = json.load(f)

    mc_path = os.path.join(
        inputs['cache'],
        'MulticutSegmentation.h5'
    )
    assert os.path.exists(mc_path)
    mc = vigra.readHDF5(mc_path, 'data')

    mc_node_path = os.path.join(
        inputs['cache'],
        'McSolverFusionMoves.h5'
    )
    assert os.path.exists(mc_node_path)
    mc_nodes = vigra.readHDF5(mc_node_path, 'data')

    print "Uniques in node-result"
    print np.unique(mc_nodes)

    print "Uniques in segmentation"
    print np.unique(mc)


def view_res(sample):

    in_file = './cremi_inputs/%s/test_files.json' % sample
    with open(in_file) as f:
        inputs = json.load(f)
    raw = vigra.readHDF5(inputs['data'][0], 'data').astype('uint32')
    pmap = vigra.readHDF5(inputs['data'][1], 'data')
    seg = vigra.readHDF5(inputs['seg'], 'data')
    gt = vigra.readHDF5(inputs['gt'], 'data')
    mc_path = os.path.join(
        inputs['cache'],
        'MulticutSegmentation.h5'
    )
    assert os.path.exists(mc_path)
    mc = vigra.readHDF5(mc_path, 'data')

    print
    print np.unique(mc)
    print

    volumina_n_layer(
        [raw, pmap, seg, gt, mc],
        ['raw', 'pmap', 'seg', 'gt', 'mc']
    )


if __name__ == '__main__':
    check_projection('sampleA_0')
    # view_res('sampleA_0')
