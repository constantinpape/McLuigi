import luigi
import os

import logging
import json

from PipelineParameter import PipelineParameter
from customLogging import config_logger
from WorkflowTasks import MulticutSegmentation, BlockwiseMulticutSegmentation
from FeatureTasks import RegionFeatures, TopologyFeatures
from MulticutSolverTasks import MCProblem

# the workflow logger
workflow_logger = logging.getLogger(__name__)

if __name__ == '__main__':

    PipelineParameter().InputFile = "../config/input_config_blkws.json"
    PipelineParameter().FeatureConfigFile = "../config/feature_config.json"
    PipelineParameter().MCConfigFile = "../config/mc_config.json"

    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)

    PipelineParameter().cache = inputs["cache"]

    config_logger(workflow_logger)
    workflow_logger.info("Starting Workflow for inputs:")
    for i, data_file in enumerate(inputs["data"]):
        workflow_logger.info("Input data nr. " + str(i) + ": " + data_file )
    workflow_logger.info("Segmentation input: " + inputs["seg"])
    workflow_logger.info("Random Forest input: " + inputs["rf"])
    workflow_logger.info("Writing cache to: " + inputs["cache"])

    # TODO get central scheduler running
    luigi.run(["--local-scheduler",
        "--PathToSeg", inputs["seg"],
        "--PathToRF", inputs["rf"]],
        main_task_cls = BlockwiseMulticutSegmentation)
        #main_task_cls = MCProblem)

    #luigi.run(["--local-scheduler",
    #    "--PathToSeg", inputs["seg"]],
    #    #"--Use2dFeatures", 'true'],
    #    main_task_cls = TopologyFeatures)
