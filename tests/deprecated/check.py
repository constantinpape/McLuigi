from volumina_viewer import volumina_n_layer
import vigra
import json
import os
import numpy as np


def check_train_cache(sample):
    cache_folder = '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/%s' % sample
    for ff in os.listdir(cache_folder):
        if ff.startswith("EdgeGroundtruth"):
            edge_file = os.path.join(cache_folder, ff)
            edges = vigra.readHDF5(edge_file, 'data')
            print ff
            print np.sum(edges), '/'
            print edges.size


def check_test_cache(sample):
    cache_folder = '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/%s' % sample
    # cache_folder = '/home/constantin/Work/home_hdd/cache/regression_tests_mcluigi/sampleA_0_test/EdgeProbabilitiesSeparate_standard.h5'

    ep_path = os.path.join(cache_folder, 'EdgeProbabilitiesSeparate_standard.h5')
    assert os.path.exists(ep_path)
    edge_probs = vigra.readHDF5(
        ep_path,
        'data'
    )
    print "Edge_Probabilities:"
    print "%f +- %f" % (np.mean(edge_probs), np.std(edge_probs))

    mc_costs = vigra.readHDF5(
        os.path.join(cache_folder, 'MulticutProblem_standard.h5'),
        'costs'
    )

    print "Costs:"
    print "%f +- %f" % (np.min(mc_costs), np.max(mc_costs))
    print "%f +- %f" % (np.mean(mc_costs), np.std(mc_costs))


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

    volumina_n_layer(
        [raw, pmap, seg, gt, mc],
        ['raw', 'pmap', 'seg', 'gt', 'mc']
    )


if __name__ == '__main__':
    # check_cache('sampleA_0_test')
    check_train_cache('sampleA_0_train')
    # view_res('sampleA_0')
