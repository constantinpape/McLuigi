import luigi
import os

import logging
import json

from PipelineParameter import PipelineParameter
from WorkflowTasks import MulticutSegmentation


if __name__ == '__main__':

    PipelineParameter().InputFile = "../config/input_config.json"
    PipelineParameter().FeatureConfigFile = "../config/feature_config.json"
    PipelineParameter().MCConfigFile = "../config/mc_config.json"

    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)

    luigi.run(["--local-scheduler",
        "--PathToSeg", inputs["seg"],
        "--PathToRF", inputs["rf"]],
        main_task_cls = MulticutSegmentation)
