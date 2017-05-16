import os
import luigi
import sys

import logging
import json

from mc_luigi import *

def sub_solver_tests():

    # PipelineParameter is a singleton class that stores most of the
    # relevant parameters for learning and inference
    ppl_parameter = PipelineParameter()

    # read the json with paths to input files
    ppl_parameter.read_input_file('./input_config_test.json')
    inputs = ppl_parameter.inputs

    # set relevant pipeline parameter for inference
    ppl_parameter.separateEdgeClassification = True # use seperate random forests for xy / z - edges
    # sub-block shapes for the block-wise multicut
    # -> chosen smaller than defaults due to small test data
    # use default values for larger data
    ppl_parameter.multicutBlockShape   = [15,256,256]
    ppl_parameter.multicutBlockOverlap = [2,10,10]

    # TODO get central scheduler running
    luigi.run(["--local-scheduler",
        "--pathToSeg", inputs["seg"],
        "--pathToClassifier", inputs["rf"]],
        TestSubSolver
    )


if __name__ == '__main__':
    sub_solver_tests()
