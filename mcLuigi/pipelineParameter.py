# implementing most pipeline parameters as they inherit from luigi.Parameter
# Pipeline Parameter are implemetned as Singleton

# singleton top class holding cache and number of cores (params that are needed by
# all Tasks, but I don't want to pass around...)

import multiprocessing
import logging

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
        self.cache = "/tmp/mc_cache"
        self.InputFile = ""
        self.FeatureConfigFile = ""
        self.MCConfigFile = ""
        self.EdgeClassifierConfigFile = ""

        # flag to switch between pipeline for anisotropic and isotropic (not implemented yet) data
        self.anisotropicPipeline = True

        ### Parameter for defect detection and handling
        # flag to switch between pipeline for normal data and defected data
        self.defectPipeline = False
        # Number of bins for the histogram of number of segments per slice
        self.nBinsSliceStatistics = 16
        # histogram threshold for the defect slicce detection
        self.binThreshold = 0 # -> 0 means we don't detect any defects ! This needs to be tuned for every ds !

        # number of threads
        self.nThreads = multiprocessing.cpu_count()
        # compression level
        self.compressionLevel = 5
        # log level
        self.logLevel = logging.INFO
        # enable using seperate classifier for xy - and z - edges
        self.separateEdgeClassification = True
        # choose betweem xgb - gradient boostig and sklearn - random forest
        self.useXGBoost = True
        # number of chunks that features are split into for out of core probability calculation
        self.nFeatureChunks = 1 # default = 1 -> in core calculation

        ### multicut and blockwise parameter
        self.globalTimeLimit = 60*60*10 # time limit in seconds / 10 hours

    # TODO range and type cheks via @property and setproperty

    # TODO don't allow setting a field that does not already exist to avoid setting incorrect parameter

    # TODO move all parameter from FeatureConfigFile, MCConfigFile and EdgeClassifierFile here

    # TODO implement load / save via serialization to json
    # include meta fields to save experiment name and time of execution
    # hash everything and put into paramstr for being able to re-identify reliably
