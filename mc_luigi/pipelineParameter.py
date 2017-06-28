# implementing most pipeline parameters as they inherit from luigi.Parameter
# Pipeline Parameter are implemetned as Singleton

# singleton top class holding cache and number of cores (params that are needed by
# all Tasks, but I don't want to pass around...)

import multiprocessing
import logging
# import os
import json


# singleton type
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PipelineParameter(object):
    __metaclass__ = Singleton

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
        # self.features = ["raw","affinitiesXY","affinitiesZ","reg"]
        self.features = ["raw", "prob", "reg"]  # for non-affinity maps replace 'affinitiesXY/Z' with prob

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
        self.wsdtMinMem    = 0
        self.wsdtMinSeg    = 75
        self.wsdtSigSeeds  = 2.6
        self.wsdtSigWeights = 0.
        self.wsdtInvert    = False

    # TODO range and type cheks via @property and setproperty

    # TODO don't allow setting a field that does not already exist to avoid setting incorrect parameter

    # TODO move all parameter from FeatureConfigFile, MCConfigFile and EdgeClassifierFile here

    # TODO implement load / save via serialization to json
    # include meta fields to save experiment name and time of execution
    # hash everything and put into paramstr for being able to re-identify reliably

    def read_input_file(self, input_file):
        with open(input_file) as f:
            inp_dict = json.load(f)
        self.inputs = inp_dict
        self.cache  = inp_dict['cache']
