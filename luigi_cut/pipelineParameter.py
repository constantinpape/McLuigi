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
        self.nThreads = 40
        self.InputFile = ""
        self.FeatureConfigFile = ""
        self.MCConfigFile = ""

    ## TODO checks...
    ## TODO this is not pythonic, overload some __ stuff instead
    #def set_cache(self, cahce):
    #    self.cache = cache

    #def set_n_threads(n_threads):
    #    self.n_threads = n_threads
