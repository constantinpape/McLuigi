import sys
import luigi
from mc_luigi import PipelineParameter
from mc_luigi.featureTasks import RegionFeatures


def region_features():
    ppl_parameter = PipelineParameter()
    ppl_parameter.read_input_file('./inputs.json')
    ppl_parameter.useN5Backend = True
    ppl_parameter.ignoreSegLabel = 0
    inputs = ppl_parameter.inputs
    inp = inputs['data'][0]
    seg = inputs["seg"]
    luigi.run(["--local-scheduler",
               "--pathToInput", inp,
               "--pathToSeg", seg],
              RegionFeatures)


def edge_features():
    pass


def affinity_xy_features():
    pass


def affinity_z_features():
    pass


if __name__ == '__main__':
    test = sys.argv[1]
    if test == 'region':
        region_features()
    elif test == 'edge':
        edge_features()
    elif test == 'affinty_xy':
        affinity_xy_features()
    elif test == 'affinty_z':
        affinity_z_features()
    else:
        assert False
