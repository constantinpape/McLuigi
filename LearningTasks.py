# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from CustomTargets import PickleTarget

import logging

import numpy as np
import vigra
#from sklearn.Ensemble import RandomForestClassifier

class ExternalRF():

    RFPath = luigi.Parameter()

    def output(self):
        return PickleTarget(RFPath)

# TODO implement learning the rf
class EdgeProbabilities(luigi.Task):

    FeatureTasks = luigi.ListParameter()
    RFPath = luigi.Parameter()

    def requires(self):
        return {"RF:", ExternalRF(self.RFPath), "Features:", [tasks for tasks in self.FeatureTasks]}

    def run(self):
        rf = self.input()["RF"].read()
        features = np.concatenate(feat for feat in self.input()["Features"].read())
        probs = rf.predict_proba(features)
        self.output().write(probs)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                "EdgeProbabilities_.h5"  )
        return HDF5Target( save_path  )


