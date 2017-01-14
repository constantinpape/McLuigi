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
        # flag to switch between pipeline for normal data and defected data
        self.defectPipeline = False
        # number of threads
        self.nThreads = multiprocessing.cpu_count()
        # compression level
        self.compressionLevel = 5
        # log level
        self.logLevel = logging.INFO

    # TODO range cheks for nThreads, compressionLevel and logLevel with @propery and set property
