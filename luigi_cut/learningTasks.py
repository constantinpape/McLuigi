# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from customTargets import PickleTarget, HDF5DataTarget
from featureTasks import get_local_features
from dataTasks import DenseGroundtruth, ExternalSegmentation, StackedRegionAdjacencyGraph

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging

import numpy as np
import nifty
import os
import time
from sklearn.ensemble import RandomForestClassifier

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

# load external random forest from pickle file
# TODO feature checks at some point...
class ExternalRandomForest(luigi.Task):

    RFPath = luigi.Parameter()

    def output(self):
        return PickleTarget(self.RFPath)


# TODO log times

class EdgeProbabilitiesFromSingleRandomForest(luigi.Task):

    randomForestTask = luigi.TaskParameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"rf" : self.randomForestTask, "features" : feature_tasks}

    def run(self):

        t_pred = time.time()

        rf = self.input()["rf"].read()
        features = np.concatenate( [feat.read() for feat in self.input()["features"]], axis = 1 )
        probs = rf.predict_proba(features)[:,1]
        t_pred = time.time() - t_pred
        workflow_logger.info("Predicted RF in: " + str(t_pred) + " s")

        self.output().write(probs)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "EdgeProbabilitiesFromSingleRandomForest.h5"  )
        return HDF5DataTarget( save_path  )


class EdgeProbabilitiesFromTwoRandomForests(luigi.Task):

    randomForestXyTask = luigi.TaskParameter()
    randomForestZTask = luigi.TaskParameter()
    pathToSeg = luigi.Parameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"rfXy" : self.randomForestXyTask, "rfZ" : self.randomForestZTask, "features" : feature_tasks, "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}

    def run(self):

        t_pred = time.time()

        rf_xy = self.input()["rfXy"].read()
        rf_z  = self.input()["rfXy"].read()

        features = np.concatenate( [feat.read() for feat in self.input()["features"]], axis = 1 )

        transitionEdge = rag.totalNumberOfInSliceEdges

        t_pred = time.time()
        probs_xy = rf_xy.predict_proba(features[transitionEdge:])[:,1]
        probs_z  = rf_z.predict_proba(features[:transitionEdge])[:,1]
        t_pred = time.time() - t_pred
        workflow_logger.info("Predicted RF in: " + str(t_pred) + " s")

        probs = np.concatenate([probs_xy, probs_z], axis = 0)

        self.output().write(probs)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "EdgeProbabilitiesFromTwoRandomForests.h5"  )
        return HDF5DataTarget( save_path  )



# TODO fuzzy mapping in nifty ?!
class EdgeGroundtruth(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        return {"gt" : DenseGroundtruth(self.pathToGt), "rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }


    def run(self):

        inp = self.input()
        gt = inp["gt"]
        gt.open()
        rag = inp["rag"].read()

        nodeGt = nifty.graph.rag.gridRagAccumulateLabels(rag, gt.get())
        uvIds = rag.uvIds()

        uGt = nodeGt[ uvIds[:,0] ]
        vGt = nodeGt[ uvIds[:,1] ]

        edgeGt = (uGt != vGt).astype(np.uint8)

        self.output().write(edgeGt)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "EdgeGroundtruth_%s.h5" % (self.pathToSeg)  )
        return HDF5DataTarget( save_path  )



class SingleRandomForestFromMultipleInputs(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToGt  = luigi.ListParameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        feature_tasks = get_local_features_for_multiinp()
        assert len(feature_tasks) == 2 * len(self.pathsToSeg)
        gts = [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in xrange(len(self.pathsToSeg))]
        return {"gts" : EdgeGroundtruth(self.pathToSeg, self.pathToGt), "features" : feature_tasks}

    def run(self):

        inp = self.input()
        gt = inp["gts"].read()
        gt = np.concatenate(gt)
        features = inp["features"]
        features_for_concat = []
        for f in xrange(len(features)/2):
            f_id0 = 2*f
            f_id1 = 2*f + 1
            features_for_concat.append( np.concatenate( features[f_id0].read(), features[f_id1].read() ), axis = 1 )
        features = np.concatenate( [feat for feat in features_for_concat, axis = 0 )

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        # TODO rf options - we shouldn't train to full purity etc. - pretty inefficient...
        # + OOB
        rf = RandomForestClassifier(n_jobs = PipelineParameter().nThreads)

        t_learn = time.time()
        rf.fit(features, gt)
        t_learn = time.time() - t_learn
        workflow_logger.info("Learned RF in: " + str(t_learn) + " s")

        self.output().write(rf)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "SingleRandomForestFromMultipleInputs.pkl"  )
        return PickleTarget(save_path)





class SingleRandomForestFromGt(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"gt" : EdgeGroundtruth(self.pathToSeg, self.pathToGt), "features" : feature_tasks}


    def run(self):

        inp = self.input()
        gt = inp["gt"].read()
        features = np.concatenate( [feat.read() for feat in inp["features"]], axis = 1 )

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        # TODO rf options - we shouldn't train to full purity etc. - pretty inefficient...
        # + OOB
        rf = RandomForestClassifier(n_jobs = PipelineParameter().nThreads)

        t_learn = time.time()
        rf.fit(features, gt)
        t_learn = time.time() - t_learn
        workflow_logger.info("Learned RF in: " + str(t_learn) + " s")

        self.output().write(rf)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "SingleRandomForestFromGt.pkl"  )
        return PickleTarget(save_path)


class RandomForestsZFromGt(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"rag" : StackedRegionAdjacencyGraph(self.pathToSeg), "gt" : EdgeGroundtruth(self.pathToSeg, self.pathToGt), "features" : feature_tasks}


    def run(self):

        inp = self.input()
        gt = inp["gt"].read()
        features = np.concatenate( [feat.read() for feat in inp["features"]], axis = 1 )
        rag = inp["rag"].read()

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        transitionEdge = rag.totalNumberOfInSliceEdges

        features_z = features[transitionEdge:]
        gt_z = gt[transitionEdge:]

        # TODO rf options - we shouldn't train to full purity etc. - pretty inefficient...
        # + OOB
        rf_z = RandomForestClassifier(n_jobs = PipelineParameter().nThreads)

        t_learn = time.time()
        rf_z.fit(features_z, gt_z)
        t_learn = time.time() - t_learn
        workflow_logger.info("Learned Random Forest on z edges in: " + str(t_learn) + " s")

        self.output().write(rf)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "RandomForestZFromGt.pkl"  )
        return PickleTarget(save_path)


class RandomForestsXyFromGt(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        feature_tasks = get_local_features()
        return {"rag" : StackedRegionAdjacencyGraph(self.pathToSeg), "gt" : EdgeGroundtruth(self.pathToSeg, self.pathToGt), "features" : feature_tasks}


    def run(self):

        inp = self.input()
        gt = inp["gt"].read()
        features = np.concatenate( [feat.read() for feat in inp["features"]], axis = 1 )
        rag = inp["rag"].read()

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        transitionEdge = rag.totalNumberOfInSliceEdges

        features_xy = features[:transitionEdge]
        gt_xy = gt[:transitionEdge]

        # TODO rf options - we shouldn't train to full purity etc. - pretty inefficient...
        # + OOB
        rf_xy = RandomForestClassifier(n_jobs = PipelineParameter().nThreads)

        t_learn = time.time()
        rf_xy.fit(features_xy, gt_xy)
        t_learn = time.time() - t_learn
        workflow_logger.info("Learned Random Forest on xy edges in: " + str(t_learn) + " s")

        self.output().write(rf)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "RandomForestXyFromGt.pkl"  )
        return PickleTarget(save_path)
