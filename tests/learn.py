import luigi
import os
import json
from mc_luigi import LearnClassifierFromGt, PipelineParameter
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    inp = args.input
    assert os.path.exists(inp)
    return inp


# run the workflow for learning the random forest on training data with groundtruth
def learn(inp):

    # PipelineParameter is a singleton class that stores most of the
    # relevant parameters for learning and inference
    ppl_parameter = PipelineParameter()

    # read the json with paths to input files
    ppl_parameter.read_input_file(inp)
    inputs = ppl_parameter.inputs

    # set the parameters
    ppl_parameter.nTrees = 500
    ppl_parameter.multicutWeightingScheme = "z"
    ppl_parameter.separateEdgeClassification = True
    ppl_parameter.multicutVerbose = True

    # TODO get central scheduler running
    luigi.run([
        "--local-scheduler",
        "--pathsToSeg", json.dumps(inputs["seg"]),
        "--pathsToGt", json.dumps(inputs["gt"])],
        LearnClassifierFromGt
    )


if __name__ == '__main__':
    inp = parse_args()
    learn(inp)
