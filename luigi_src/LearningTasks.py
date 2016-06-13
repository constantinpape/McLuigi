# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from CustomTargets import PickleTarget, HDF5Target
from PipelineParameter import PipelineParameter
from FeatureTasks import get_local_features

import logging

import numpy as np
import vigra
import os
#from sklearn.Ensemble import RandomForestClassifier

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
        rf = self.input()["RF"].read()
        features = np.concatenate( [feat.read() for feat in self.input()["Features"]], axis = 1 )
        probs = rf.predict_proba(features)[:,1]
        self.output().write(probs)

    def output(self):
        # TODO more expressive caching name
        save_path = os.path.join( PipelineParameter().cache,
                "EdgeProbabilities.h5"  )
        return HDF5Target( save_path  )

# TODO implement learning the rf
