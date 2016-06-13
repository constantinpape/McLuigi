import luigi
import os

import logging
import json

from PipelineParameter import PipelineParameter
from WorkflowTasks import MulticutSegmentation, BlockwiseMulticutSegmentation
from FeatureTasks import RegionFeatures, TopologyFeatures
from MulticutSolverTasks import MCProblem


if __name__ == '__main__':

    PipelineParameter().InputFile = "../config/input_config_blkws.json"
    PipelineParameter().FeatureConfigFile = "../config/feature_config.json"
    PipelineParameter().MCConfigFile = "../config/mc_config.json"

    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)

    PipelineParameter().cache = inputs["cache"]

    luigi.run(["--local-scheduler",
        "--PathToSeg", inputs["seg"],
        "--PathToRF", inputs["rf"]],
        main_task_cls = BlockwiseMulticutSegmentation)
        #main_task_cls = MCProblem)

    #luigi.run(["--local-scheduler",
    #    "--PathToSeg", inputs["seg"]],
    #    #"--Use2dFeatures", 'true'],
    #    main_task_cls = TopologyFeatures)
