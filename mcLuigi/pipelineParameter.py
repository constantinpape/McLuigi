# implementing most pipeline parameters as they inherit from luigi.Parameter
# Pipeline Parameter are implemetned as Singleton

# singleton top class holding cache and number of cores (params that are needed by
# all Tasks, but I don't want to pass around...)

import luigi

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
        # TODO this should also be read from some json...
        self.cache = "/tmp/mc_cache"

        import multiprocessing
        self.nThreads = multiprocessing.cpu_count()

        self.InputFile = ""
        self.FeatureConfigFile = ""
        self.MCConfigFile = ""
        self.EdgeClassifierConfigFile = ""

        # flag to switch between pipeline for normal data and defected data
        self.defectPipeline = False

        # compression level
        self.compressionLevel = 5

    ## TODO checks...
    ## TODO this is not pythonic, @propery and set property instead
    #def set_cache(self, cahce):
    #    self.cache = cache

    #def set_n_threads(n_threads):
    #    self.n_threads = n_threads
