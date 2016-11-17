import luigi
import os

import logging
import json

from workflowTasks import MulticutSegmentation, BlockwiseMulticutSegmentation
from featureTasks import EdgeFeatures, RegionFeatures
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentationLabeled
from learningTasks import SingleRandomForestFromGt

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

# the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

if __name__ == '__main__':

    PipelineParameter().InputFile = "config/input_config.json"
    PipelineParameter().FeatureConfigFile = "config/feature_config.json"
    PipelineParameter().MCConfigFile = "../config/mc_config.json"

    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)

    PipelineParameter().cache = inputs["cache"]

    workflow_logger.info("Starting Workflow for inputs:")
    for i, data_file in enumerate(inputs["data"]):
        workflow_logger.info("Input data nr. " + str(i) + ": " + data_file )
    workflow_logger.info("Segmentation input: " + inputs["seg"])
    #workflow_logger.info("Random Forest input: " + inputs["rf"])
    workflow_logger.info("Writing cache to: " + inputs["cache"])

    # TODO get central scheduler running
    luigi.run(["--local-scheduler",
        "--pathToSeg", inputs["seg"],
        #"--pathToGt", inputs["gt"]],
        #"--numberOfLevels", 1,
        "--pathToRF", inputs["rf"]],
        main_task_cls = MulticutSegmentation)
        #main_task_cls = BlockwiseMulticutSegmentation)
        #main_task_cls = SingleRandomForestFromGt)
