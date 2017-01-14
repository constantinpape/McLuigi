import logging
import time
import numpy as np
import hashlib

from functools import wraps

# call this function to configure the logger
# change the log file / log level if necessary
def config_logger(logger, level = logging.INFO):

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
    def wrapper(*args,**kwargs):
        self = args[0]
        name = self.__class__.__name__
        # for some reason this can get called multiple times (though the actual run function is always only executed once)
        # that's why we have to make sure that this is only logged once to not clutter the log
        param_names = self.get_param_names()
        param_values = [eval("self.%s" % param_name) for param_name in param_names]
        hash_str = "".join([name]+param_names+[str(param_val) for param_val in param_values])
        hash_str = hashlib.md5(hash_str.encode())
        hash_str = hash_str.hexdigest()
        if hash_str not in decorator_call_list:
            workflow_logger.info("%s: run called with input parameters:" % name)
            for i, param_name in enumerate(param_names):
                workflow_logger.info("%s: %s: %s" % (name ,param_name, str(param_values[i])))
            decorator_call_list.append(hash_str)
        t_run = time.time()
        ret = function(*args,**kwargs)
        workflow_logger.info("%s: run finished in %f s" % (name,time.time() - t_run) )
        return ret
    return wrapper


# numpy.replace: replcaces the values in array according to dict
def replace(array, dict_like):
    replace_keys, replace_vals = np.array(list(zip( *sorted(dict_like.items() ))))
    indices = np.digitize(array, replace_keys, right = True)
    return replace_vals[indices]
