from __future__ import division, print_function

import logging
import time
import hashlib
import numpy as np

from functools import wraps


# call this function to configure the logger
# change the log file / log level if necessary
def config_logger(logger, level=logging.INFO):

    # for now we have to change this here, until we have better configuration handling
    handler = logging.FileHandler('luigi_workflow.log')
    handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(handler)


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

decorator_call_list = []


# wrapper for the run method of luigi.Task, that logs the runtime and the input parameter
def run_decorator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        self = args[0]
        name = self.__class__.__name__
        param_names = self.get_param_names()
        param_values = [getattr(self, param_name) for param_name in param_names]

        # for some reason this can get called multiple times
        # (though the actual run function is always only executed once)
        # that's why we have to make sure that this is only logged once to not clutter the log
        hash_str = "".join([name] + param_names + [str(param_val) for param_val in param_values])
        hash_str = hashlib.md5(hash_str.encode())
        hash_str = hash_str.hexdigest()
        if hash_str not in decorator_call_list:
            workflow_logger.info("%s: run called with input parameters:" % name)
            for i, param_name in enumerate(param_names):
                workflow_logger.info("%s: %s: %s" % (name, param_name, str(param_values[i])))
            decorator_call_list.append(hash_str)

        t_run = time.time()
        ret = function(*args, **kwargs)
        workflow_logger.info("%s: run finished in %f s" % (name, time.time() - t_run))
        return ret
    return wrapper


def get_replace_slices(defected_slices, shape):

    # find consecutive slices with defects
    consecutive_defects = np.split(defected_slices, np.where(np.diff(defected_slices) != 1)[0] + 1)
    # find the replace slices for defected slices
    replace_slice = {}
    for consec in consecutive_defects:
        if len(consec) == 1:
            z = consec[0]
            replace_slice[z] = z - 1 if z > 0 else 1
        elif len(consec) == 2:
            z0, z1 = consec[0], consec[1]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 2
            replace_slice[z1] = z1 + 1 if z1 < shape[0] - 1 else z1 - 2
        elif len(consec) == 3:
            z0, z1, z2 = consec[0], consec[1], consec[2]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 3
            replace_slice[z1] = z1 - 2 if z1 > 1 else 3
            replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
        elif len(consec) == 3:
            z0, z1, z2, z3 = consec[0], consec[1], consec[2], consec[3]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 4
            replace_slice[z1] = z1 - 2 if z1 > 1 else 4
            replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
            replace_slice[z3] = z3 + 2 if z3 < shape[0] - 1 else z3 - 4
        else:
            raise RuntimeError(
                "Postprocessing is not implemented for more than 4 consecutively defected slices. Clean your data!"
            )

    return replace_slice
