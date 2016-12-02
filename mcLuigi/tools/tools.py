import logging
import numpy as np

# call this function to configure the logger
# change the log file / log level if necessary
def config_logger(logger):

    level = logging.INFO
    #level = logging.DEBUG

    # for now we have to change this here, until we have better configuration handling
    handler = logging.FileHandler('luigi_workflow.log')
    handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(handler)

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)



# numpy.replace: replcaces the values in array according to dict
def replace(array, dict_like):
    replace_keys, replace_vals = np.array(list(zip( *sorted(dict_like.items() ))))
    indices = np.digitize(array, replace_keys, right = True)
    return replace_vals[indices]
