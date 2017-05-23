import luigi
import os
from mc_luigi import BlockwiseMulticutSegmentation, MulticutSegmentation, PipelineParameter
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    inp = args.input
    assert os.path.exists(inp)
    return inp


def mc(inp, blockwise_inference=False):

    ppl_parameter = PipelineParameter()
    # read the json with paths to input files
    ppl_parameter.read_input_file(inp)
    inputs = ppl_parameter.inputs
    # set the parameters
    ppl_parameter.nTrees = 500
    ppl_parameter.multicutWeightingScheme = "z"
    ppl_parameter.separateEdgeClassification = True
    ppl_parameter.multicutVerbose = True
    ppl_parameter.multicutBlockShape = [50, 512, 512]
    ppl_parameter.multicutBlockOverlap = [5, 50, 50]
    ppl_parameter.nFeatureChunks = 1
    n_levels = 1

    if blockwise_inference:
        luigi.run(
            ["--local-scheduler",
             "--pathToSeg", inputs["seg"],
             "--pathToClassifier", inputs["rf"],
             "--numberOfLevels", str(n_levels)],
            BlockwiseMulticutSegmentation
        )
    else:
        luigi.run(
            ["--local-scheduler",
             "--pathToSeg", inputs["seg"],
             "--pathToClassifier", inputs["rf"]],
            MulticutSegmentation
        )


if __name__ == '__main__':
    inp = parse_args()
    mc(inp)
