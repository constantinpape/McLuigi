import os
import luigi
import sys

import logging
import json

from mc_luigi import *

# configure the logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


def print_instructions():
    print "Input files not found."
    print "Either download and unzip the test data from:"
    print "https://drive.google.com/open?id=0B4_sYa95eLJ1ek8yMWozTzhBbGM"
    print "or provide your own data"
    print "(check input_config_train.json / input_config_test.json for the input format)"
    sys.exit()


def check_inputs(inputs, check_rf = False):
    for key, paths in inputs.iteritems():
        if key == 'rf' and not check_rf:
            continue
        if key == 'cache':
            continue
        if isinstance(paths, list):
            for path in paths:
                if not os.path.exists(path):
                    print_instructions()
        else:
            if not os.path.exists(paths):
                print_instructions()


# run the workflow for learning the random forest on training data with groundtruth
def learning():

    # PipelineParameter is a singleton class that stores most of the
    # relevant parameters for learning and inference
    ppl_parameter = PipelineParameter()

    # read the json with paths to input files
    ppl_parameter.read_input_file('./input_config_train.json')
    inputs = ppl_parameter.inputs
    check_inputs(inputs)

    # set relevant pipeline parameter for learning
    ppl_parameter.separateEdgeClassification = True # use seperate random forests for xy / z - edges
    ppl_parameter.nTrees = 100 # number of trees used in rf

    # TODO get central scheduler running
    luigi.run([
        "--local-scheduler",
        "--pathsToSeg", json.dumps([inputs["seg"]]),
        "--pathsToGt",  json.dumps([inputs["gt" ]])],
        LearnClassifierFromGt
    )


def inference(blockwise_inference = True):

    # PipelineParameter is a singleton class that stores most of the
    # relevant parameters for learning and inference
    ppl_parameter = PipelineParameter()

    # read the json with paths to input files
    ppl_parameter.read_input_file('./input_config_test.json')
    inputs = ppl_parameter.inputs
    check_inputs(inputs, check_rf = True)

    # set relevant pipeline parameter for inference
    ppl_parameter.separateEdgeClassification = True # use seperate random forests for xy / z - edges
    # sub-block shapes for the block-wise multicut
    # -> chosen smaller than defaults due to small test data
    # use default values for larger data
    ppl_parameter.multicutBlockShape   = [15,256,256]
    ppl_parameter.multicutBlockOverlap = [2,10,10]

    # number of mergign levels in block-wise multicut
    # -> increase if the final multicut for merging the global reduced
    # problem takes too long for converging
    n_levels = 1

    if blockwise_inference:
        # TODO get central scheduler running
        luigi.run(["--local-scheduler",
            "--pathToSeg", inputs["seg"],
            "--pathToClassifier", inputs["rf"],
            "--numberOfLevels", str(n_levels)],
            BlockwiseMulticutSegmentation
        )
    else:
        # TODO get central scheduler running
        luigi.run(["--local-scheduler",
            "--pathToSeg", inputs["seg"],
            "--pathToClassifier", inputs["rf"]],
            MulticutSegmentation
        )


if __name__ == '__main__':
    # first call learning, then inference (need to execute 'python example_experiments.py' two seperate times)
    #learning()
    inference()
