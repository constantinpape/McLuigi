from __future__ import print_function, division

# implementing most pipeline parameters as they inherit from luigi.Parameter
# Pipeline Parameter are implemetned as Singleton

# singleton top class holding cache and number of cores (params that are needed by
# all Tasks, but I don't want to pass around...)

import multiprocessing
import logging
import os
import json


# singleton type
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# FIXME this is not python 2 compatible
class PipelineParameter(object, metaclass=Singleton):
    #__metaclass__ = Singleton

    def __init__(self):

        # Input Files
        self.inputs = {}
        self.cache  = ""

        # flag to switch between pipeline for anisotropic and isotropic (not implemented yet) data
        self.anisotropicPipeline = True

        # Parameter for defect detection and handling
        # flag to switch between pipeline for normal data and defected data
        self.defectPipeline = False
        # Number of bins for the histogram of number of segments per slice
        self.nBinsSliceStatistics = 16
        # histogram threshold for the defect slicce detection
        self.binThreshold = 2  # -> 0 means we don't detect any defects ! This needs to be tuned for every ds !

        # feature string
        # FIXME in the current form z-affnity are toxic for defects!
        # for non-affinity maps replace 'affinitiesXY/Z' with prob
        # self.features = ["raw", "affinitiesXY", "affinitiesZ", "reg"]
        self.features = ["raw", "prob", "reg"]
        self.zAffinityDirection = 2 # encoding of z-affinities: 1 -> slice z has affinties to z+1, 2 -> z+1 has affinities to z

        # path to neural network snapshots
        self.netWeightsPath = ''
        self.netArchitecturePath = ''
        self.netGpuId = 0  # id of gpu to be used

        # ignore label for edge groundtruth
        self.haveIgnoreLabel = False
        self.ignoreLabel = -1

        # number of threads
        self.nThreads = multiprocessing.cpu_count()
        # compression level
        self.compressionLevel = 5
        # log level
        self.logLevel = logging.INFO
        # enable using seperate classifier for xy - and z - edges
        self.separateEdgeClassification = True
        # number of chunks that features are split into for out of core probability calculation
        self.nFeatureChunks = 1  # default = 1 -> in core calculation
        # rf params: number of trees and max depth
        self.nTrees = 500
        self.maxDepth = 12

        # multicut and blockwise parameter
        self.multicutVerbose = 0
        self.multicutSigmaFusion = 10.
        self.multicutNumParallelProposals = 1
        self.multicutNumThreadsFusion = 1
        self.multicutNumFuse = 0  # we don't fuse by default, because this turns out to be slower for large data
        self.multicutBeta = 0.5
        self.multicutWeightingScheme = "z"
        self.multicutWeight = 16
        self.multicutSeedFraction = 0.05
        self.multicutNumIt = 2500
        self.multicutNumItStop = 20
        self.multicutBlockShape = [50, 512, 512]
        self.multicutBlockOverlap = [5, 20, 20]
        self.multicutGlobalTimeLimit = 60 * 60 * 10  # time limit in seconds / 10 hours
        self.multicutNThreadsGlobal = multiprocessing.cpu_count()
        self.multicutNumItStopGlobal = 12
        self.multicutSeedFractionGlobal = 1e-5

        self.subSolverType = 'fm-ilp'
        self.globalSolverType = 'fm-kl'

        # parameters for segmentation tasks / wsdt
        self.wsdtThreshold = .2
        self.wsdtMinSeg    = 75
        self.wsdtSigSeeds  = 2.6
        self.wsdtInvert    = False

        # for BlockWiseStitchingSolver
        self.overlapThreshold = .5

    # TODO range and type cheks via @property and setproperty

    # TODO don't allow setting a field that does not already exist to avoid setting incorrect parameter

    # TODO move all parameter from FeatureConfigFile, MCConfigFile and EdgeClassifierFile here

    # TODO implement load / save via serialization to json
    # include meta fields to save experiment name and time of execution
    # hash everything and put into paramstr for being able to re-identify reliably

    # read the inputs, depending of number and types of inputs
    # decide the appropriate data tasks:
    # no segmentation -> assume we don't have probabilities either
    # and produce pmaps as well as segmentation
    def read_input_file(self, input_file):
        with open(input_file) as f:
            inp_dict = json.load(f)

        self.cache  = inp_dict['cache']
        if not os.path.exists(self.cache):
            os.mkdir(self.cache)

        # check if we have an over-segmentation already
        # if not we will schedule probability map prediction and watershed task
        if 'seg' not in inp_dict:
            data_list = inp_dict['data']
            if isinstance(data_list, str):
                n_inp = 1
                data_list = [data_list]
            else:
                assert isinstance(data_list, list)
                n_inp = len(data_list)

            # append the affinity maps that will be predicted to inputs
            new_data_list = []
            # append the watersheds that will be produced to the inputs
            seg_list = []
            for inp in range(n_inp):

                raw_path = data_list[inp]
                new_data_list.append(raw_path)
                raw_prefix = os.path.split(raw_path)[1][:-3]

                affinity_folder = os.path.join(self.cache, '%s_affinities' % raw_prefix)

                # xy-affinities
                affinity_xy_path = os.path.join(affinity_folder, '%s_affinities_xy.h5' % raw_prefix)
                new_data_list.append(affinity_xy_path)

                # z-affinities
                affinity_z_path = os.path.join(affinity_folder, '%s_affinities_z.h5' % raw_prefix)
                new_data_list.append(affinity_z_path)

                # wsdt segmentation
                seg_list.append(
                    os.path.join(self.cache, 'WsdtSegmentation_%s_affinities_xy.h5' % raw_prefix)
                )

            inp_dict['data'] = new_data_list
            inp_dict['seg'] = seg_list

        self.inputs = inp_dict
