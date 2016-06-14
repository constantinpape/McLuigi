# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from customTargets import PickleTarget, HDF5Target
from featureTasks import get_local_features

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging

import numpy as np
import vigra
import os
import time
#from sklearn.Ensemble import RandomForestClassifier

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

class ExternalRF(luigi.Task):

    RFPath = luigi.Parameter()

    def output(self):
        return PickleTarget(self.RFPath)

# TODO no checks are done here
class EdgeProbabilitiesFromExternalRF(luigi.Task):

    RFPath = luigi.Parameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"RF" : ExternalRF(self.RFPath), "Features" : feature_tasks}

    def run(self):

        t_pred = time.time()

        rf = self.input()["RF"].read()
        features = np.concatenate( [feat.read() for feat in self.input()["Features"]], axis = 1 )
        probs = rf.predict_proba(features)[:,1]

        t_pred = time.time() - t_pred
        workflow_logger.info("Predicted RF in: " + str(t_pred) + " s")

        self.output().write(probs)

    def output(self):
        # TODO more expressive caching name
        save_path = os.path.join( PipelineParameter().cache,
                "EdgeProbabilities.h5"  )
        return HDF5Target( save_path  )

# TODO implement learning the rf
