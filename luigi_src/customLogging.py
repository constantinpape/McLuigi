from PipelineParameter import PipelineParameter

import logging

# configure the logger
def config_logger(logger):

    # for now we have to change this here, until we have better configuration handling
    handler = logging.FileHandler('luigi_workflow.log')
    handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
