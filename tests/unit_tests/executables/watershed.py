import sys

import luigi
from mc_luigi import PipelineParameter
from mc_luigi import WsdtSegmentation


def wsdt_default():
    ppl_parameter = PipelineParameter()
    ppl_parameter.useN5Backend = True
    ppl_parameter.read_input_file('./inputs.json')
    ppl_parameter.nThreads = 8
    ppl_parameter.wsdtInvert = True
    inp = ppl_parameter.inputs['data'][1]

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", inp,
               "--keyToProbabilities", "data"],
              WsdtSegmentation)


def wsdt_nominseg():
    ppl_parameter = PipelineParameter()
    ppl_parameter.useN5Backend = True
    ppl_parameter.read_input_file('./inputs.json')
    ppl_parameter.wsdtMinSeg = 0
    ppl_parameter.nThreads = 8
    ppl_parameter.wsdtInvert = True
    inp = ppl_parameter.inputs['data'][1]

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", inp,
               "--keyToProbabilities", "data"],
              WsdtSegmentation)


def wsdt_masked():
    ppl_parameter = PipelineParameter()
    ppl_parameter.useN5Backend = True
    ppl_parameter.read_input_file('./inputs.json')
    inp = ppl_parameter.inputs['data'][1]
    ppl_parameter.nThreads = 8
    ppl_parameter.wsdtInvert = True
    mask = ppl_parameter.inputs['mask']

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", inp,
               "--keyToProbabilities", "data",
               "--pathToMask", mask],
              WsdtSegmentation)


def wsdt_masked_nominseg():
    ppl_parameter = PipelineParameter()
    ppl_parameter.useN5Backend = True
    ppl_parameter.read_input_file('./inputs.json')
    inp = ppl_parameter.inputs['data'][1]
    ppl_parameter.wsdtMinSeg = 0
    ppl_parameter.nThreads = 8
    ppl_parameter.wsdtInvert = True
    mask = ppl_parameter.inputs['mask']

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", inp,
               "--keyToProbabilities", "data",
               "--pathToMask", mask],
              WsdtSegmentation)


if __name__ == '__main__':
    test = sys.argv[1]
    if test == 'default':
        wsdt_default()
    elif test == 'nominseg':
        wsdt_nominseg()
    elif test == 'masked':
        wsdt_masked()
    elif test == 'masked_nominseg':
        wsdt_masked_nominseg()
    else:
        assert False
