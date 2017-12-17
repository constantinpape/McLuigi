import sys
import luigi
from mc_luigi import PipelineParameter
from mc_luigi.featureTasks import RegionFeatures


def region_features():
    pass


if __name__ == '__main__':
    test = sys.argv[1]
    if test == 'region':
        region_features()
    elif test == 'edge':
        pass
    else:
        assert False
