# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from taskSelection import get_local_features,get_local_features_for_multiinp
from customTargets import PickleTarget, HDF5DataTarget, HDF5VolumeTarget
from dataTasks import DenseGroundtruth, ExternalSegmentation, StackedRegionAdjacencyGraph
from defectHandlingTasks import ModifiedAdjacency

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging

import numpy as np
import nifty
import os
import time
import json
import cPickle as pickle

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

try:
    import xgboost as xgb
    have_xgb = True
except ImportError:
    workflow_logger.info("xgboost is not installed!")
    have_xgb = False

try:
    from sklearn.ensemble import RandomForestClassifier
    have_rf = True
except ImportError:
    workflow_logger.info("sklearn is not installed!")
    have_rf = False

if (not have_xgb) and (not have_rf):
    raise ImportError("Could not import xgboost and sklearn!")


# TODO log important params like n_rounds and n_trees
# wrapper for xgboost / sklearn.rf classifier
class EdgeClassifier(object):

    def __init__(self):
        self.use_xgb = PipelineParameter().useXGBoost
        if self.use_xgb:
            assert have_xgb, "Trying to use xgb, which could not be imorted."
        else:
            assert have_rf, "Trying to use sklearn, which could not be imorted."
        self.cf = None

    def train(self, x, y):
        assert x.shape[0] == y.shape[0]
        if self.use_xgb:
            self._train_xgb(x,y)
        else:
            self._train_rf(x,y)

    def _train_xgb(self,x,y):
        # read the classifier parameter
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)
        silent = 1
        if cf_config['verbose']:
            silent = 0
        # TODO use cf_config.get(key, default) instead of cf_config[key] with sensible defaults
        xgb_params = { 'objective' : 'multi:softprob', 'num_class' : 2, # these two should not be changed
            'eta' : cf_config['xgbEta'],
            'max_delta_step' : cf_config['xgbMaxDeltaStep'] ,
            'colsample_by_tree' : cf_config['xgbColsample'],
            'subsample' : cf_config['xgbSubsample'],
            'silent' : silent }
        n_rounds = cf_config['xgbNumRounds']
        xgb_mat = xgb.DMatrix(x, label = y)
        self.cf = xgb.train(xgb_params, xgb_mat, n_rounds)

    def _train_rf(self,x,y):
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)
        verbose = 0
        if cf_config['verbose']:
            verbose = 2
        # TODO use cf_config.get(key, default) instead of cf_config[key] with sensible defaults
        self.cf = RandomForestClassifier(n_jobs = cf_config['sklearnNJobs'],
            n_estimators     = cf_config['sklearnNtrees'],
            verbose          = verbose,
            max_depth        = cf_config['sklearnMaxDepth'],
            min_samples_leaf = cf_config['sklearnMinSamplesLeaf'],
            bootstrap        = cf_config['sklearnBootstrap'])
        self.cf.fit(x, y)

    def predict(self, x):
        assert self.cf is not None, "Need to load or train a classifier before predicting."
        if self.use_xgb:
            return self._predict_xgb(x)
        else:
            return self._predict_rf(x)

    def _predict_xgb(self, x):
        xgb_mat = xgb.DMatrix(x)
        return self.cf.predict(xgb_mat)[:,1]

    def _predict_rf(self, x):
        return self.cf.predict_proba(x)[:,1]

    def load(self, path):
        assert os.path.exists(path), path
        # TODO assert that we have the right type - xgb vs rf
        with open(path) as f:
            self.cf = pickle.load(f)

    def get_classifier(self):
        assert self.cf is not None, "Need to load or train a classifier first."
        return self.cf


class EdgeProbabilities(luigi.Task):

    pathToSeg              = luigi.Parameter()
    pathsToClassifier      = luigi.ListParameter()

    def requires(self):
        nClassifiers = 2 if PipelineParameter().separateEdgeClassification else 1
        assert len(self.pathsToClassifier) == nClassifiers, "%i, %i" % (self.pathsToClassifier, nClassifiers)
        return_tasks ={"features" : get_local_features(), "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return return_tasks

    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp["rag"].read()
        feature_tasks = inp["features"]
        for feat in feature_tasks:
            feat.open()

        if PipelineParameter().defectPipeline:
            nEdges = inp["modified_adjacency"].read("n_edges_modified")
            assert nEdges > 0, str(nEdges)
            workflow_logger.info("EdgeProbabilities: for defect corrected edges. Total number of edges: %i" % nEdges)
        else:
            nEdges = rag.numberOfEdges
            workflow_logger.info("EdgeProbabilities: Total number of edges: %i" % nEdges)

        out  = self.output()
        out.open([nEdges],[min(262144,nEdges)]) # 262144 = chunk size

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges separately.")
            self._predict_separate(rag, feature_tasks, out)
        else:
            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges jointly out of core.")
                self._predict_joint_out_of_core(rag, feature_tasks, out)
            else:
                workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges jointly in core.")
                self._predict_joint_in_core(rag, feature_tasks, out)

        for feat in feature_tasks:
            feat.close()
        out.close()


    def _predict_joint_in_core(self, rag, feature_tasks, out):
        classifier = EdgeClassifier()
        classifier.load(self.pathsToClassifier[0])
        features = np.concatenate( [feat.read([0,0],feat.shape) for feat in feature_tasks], axis = 1 )
        for feat in feature_tasks:
            feat.close()
        probs = classifier.predict(features)
        out.write([0L],probs)


    def _predict_joint_out_of_core(self, rag, feature_tasks, out):
        classifier = EdgeClassifier()
        classifier.load(self.pathsToClassifier[0])
        nEdges = rag.numberOfEdges

        nSubFeats = PipelineParameter().nFeatureChunks
        assert nSubFeats > 1, nSubFeats
        for subFeats in xrange(nSubFeats):
            print subFeats, '/', nSubFeats
            featIndexStart = int(float(subFeats)/nSubFeats * nEdges)
            featIndexEnd   = int(float(subFeats+1)/nSubFeats * nEdges)
            if subFeats == nSubFeats:
                featIndexEnd = nEdges
            nEdgesSub = featIndexEnd - featIndexStart

            featuresSub = np.concatenate( [feat.read([featIndexStart,0],[featIndexEnd,feat.shape[1]]) for feat in feature_tasks], axis = 1 )
            assert featuresSub.shape[0] == nEdgesSub
            probsSub = classifier.predict(featuresSub)
            out.write([featIndexStart],probsSub)


    def _predict_separate(self, rag, feature_tasks, out):

        if PipelineParameter().defectPipeline:
            nEdges = self.input()["modified_adjacency"].read("n_edges_modified")
            assert nEdges > 0
            nXYEdges = rag.totalNumberOfInSliceEdges
            nZEdges  = nEdges - nXYEdges
        else:
            nEdges   = rag.numberOfEdges
            nXYEdges = rag.totalNumberOfInSliceEdges
            nZEdges  = nEdges - nXYEdges

        nFeatsXY = 0
        nFeatsZ = 0
        for feat in feature_tasks:
            if feat.shape[0] == nEdges:
                nFeatsXY += feat.shape[1]
                nFeatsZ += feat.shape[1]
            elif feat.shape[0] == nXYEdges:
                nFeatsXY += feat.shape[1]
            elif feat.shape[0] == nZEdges:
                nFeatsZ += feat.shape[1]
            else:
                raise RuntimeError("Number of features: %i does not match number of edges (Total: %i, XY: %i, Z: %i )" % (feat.shape[0],nEdges,nXYEdges,nZEdges))

        for i, feat_type in enumerate( ['xy', 'z'] ):
            print "Predicting", feat_type, "features"
            classifier = EdgeClassifier()
            classifier.load(self.pathsToClassifier[i])

            nEdgesType = nXYEdges
            nEdgesNonType = nZEdges
            startType  = 0
            endType    = nXYEdges
            nFeatsTotal = nFeatsXY
            if feat_type == 'z':
                nEdgesType = nZEdges
                nEdgesNonType = nXYEdges
                startType = nXYEdges
                endType = nEdges
                nFeatsTotal = nFeatsZ

            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info("EdgeProbabilities: running separate prediction for %s edges out of core" % feat_type)
                self._predict_separate_out_of_core(classifier,
                        feature_tasks,
                        out,
                        nEdges,
                        nEdgesType,
                        nEdgesNonType,
                        startType,
                        endType,
                        nFeatsTotal)
            else:
                workflow_logger.info("EdgeProbabilities: running separate prediction for %s edges in core" % feat_type)
                self._predict_separate_in_core(classifier,
                        feature_tasks,
                        out,
                        nEdges,
                        nEdgesType,
                        nEdgesNonType,
                        startType,
                        endType,
                        nFeatsTotal)


    # In core prediction for edge type
    def _predict_separate_in_core(self,
            classifier,
            feature_tasks,
            out,
            nEdges,
            nEdgesType,
            nEdgesNonType,
            startType,
            endType,
            nFeatsTotal):

            # In core prediction
            featuresType = np.zeros( (nEdgesType, nFeatsTotal), dtype = 'float32' )
            featOffset = 0

            for ii, feat in enumerate(feature_tasks):
                nFeats = feat.shape[0]
                if nFeats == nEdges:
                    #print "Feat common for xy and z"
                    featuresType[:,featOffset:featOffset+feat.shape[1]] = feat.read(
                        [long(startType),0L], [long(endType),long(feat.shape[1])])
                elif nFeats == nEdgesType:
                    #print "Only", feat_type, "feature"
                    featuresType[:,featOffset:featOffset+feat.shape[1]] = feat.read(
                            [0L,0L],
                            feat.shape
                        )
                elif nFeats == nEdgesNonType:
                    #print "Not", feat_type, "feature"
                    continue
                else:
                    raise RuntimeError("Number of features: %i does not match number of edges (Total: %i, XY: %i, Z: %i )" % (nFeats,nEdges,nXYEdges,nZEdges))
                featOffset += feat.shape[1]

            print "Features loaded, starting prediction"
            out.write([long(startType)], classifier.predict(featuresType))


    # Out of core prediction for edge type
    def _predict_separate_out_of_core(self,
            classifier,
            feature_tasks,
            out,
            nEdges,
            nEdgesType,
            nEdgesNonType,
            startType, endType,
            nFeatsTotal):

        nSubFeats = PipelineParameter().nFeatureChunks
        assert nSubFeats > 1, str(nSubFeats)
        for subFeats in xrange(nSubFeats):
            print subFeats, '/', nSubFeats
            featIndexStart = int(float(subFeats)/nSubFeats * nEdgesType)
            featIndexEnd   = int(float(subFeats+1)/nSubFeats * nEdgesType)
            if subFeats == nSubFeats:
                featIndexEnd = nEdgesType
            nEdgesSub = featIndexEnd - featIndexStart

            featuresType = np.zeros( (nEdgesSub, nFeatsTotal), dtype = 'float32' )
            featOffset = 0

            for ii, feat in enumerate(feature_tasks):
                nFeats = feat.shape[0]
                if nFeats == nEdges:
                    #print "Feat common for xy and z"
                    readStart = featIndexStart + startType
                    readEnd   = featIndexEnd + startType
                    featuresType[:,featOffset:featOffset+feat.shape[1]] = feat.read(
                            [long(readStart),0L],
                            [long(readEnd),long(feat.shape[1])]
                        )
                elif nFeats == nEdgesType:
                    #print "Only", feat_type, "feature"
                    featuresType[:,featOffset:featOffset+feat.shape[1]] = feat.read(
                            [long(featIndexStart),0L],
                            [long(featIndexEnd), long(feat.shape[1])]
                        )
                elif nFeats == nEdgesNonType:
                    #print "Not", feat_type, "feature"
                    continue
                else:
                    raise RuntimeError("Number of features: %i does not match number of edges (Total: %i, XY: %i, Z: %i )" % (nFeats,nEdges,nXYEdges,nZEdges))

                featOffset += feat.shape[1]

            # TODO do we really not need this concatenation?
            #featuresType = np.concatenate( featuresType, axis = 1 )
            print "Features loaded, starting prediction"

            readStart = long(featIndexStart + startType)
            out.write([readStart], classifier.predict(featuresType))

    def output(self):
        save_path = os.path.join(PipelineParameter().cache, "EdgeProbabilities%s.h5" % ("Separate" if PipelineParameter().separateEdgeClassification else "Joint",))
        return HDF5VolumeTarget(save_path, 'float32')


# TODO fuzzy mapping in nifty ?!
class EdgeGroundtruth(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        return {"gt" : DenseGroundtruth(self.pathToGt), "rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }

    @run_decorator
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

        assert (np.unique(edgeGt) == np.array([0,1])).all(), str(np.unique(edgeGt))
        assert edgeGt.shape[0] == uvIds.shape[0]

        self.output().write(edgeGt)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "EdgeGroundtruth_%s.h5" % (segFile,)  )
        return HDF5DataTarget( save_path  )


class LearnClassifierFromGt(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToGt  = luigi.ListParameter()
    learnXYOnly  = luigi.BoolParameter(default = False)
    learnZOnly   = luigi.BoolParameter(default = False)

    def requires(self):
        assert not (self.learnXYOnly and self.learnZOnly)
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        n_inputs = len(self.pathsToSeg)

        if n_inputs == 1:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from single input")
            feature_tasks = get_local_features(self.learnXYOnly, self.learnZOnly)
            return_tasks = {"gt" : EdgeGroundtruth(self.pathsToSeg[0], self.pathsToGt[0]),
                            "features" : feature_tasks,
                            "rag" : StackedRegionAdjacencyGraph(self.pathsToSeg[0])}
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from %i inputs" % n_inputs)
            feature_tasks = get_local_features_for_multiinp(self.learnXYOnly, self.learnZOnly)
            assert len(feature_tasks) == n_inputs
            return_tasks = {"gt" : [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in xrange(n_inputs)],
                            "features" : feature_tasks,
                            "rag" : [StackedRegionAdjacencyGraph(segp) for segp in self.pathsToSeg]}
        return return_tasks

    @run_decorator
    def run(self):
        inp = self.input()
        gt = inp["gt"]
        feature_tasks = inp["features"]
        rag = inp["rag"]

        n_inputs = len(self.pathsToSeg)

        if n_inputs > 1:
            n_inputs = len(gt)
            assert n_inputs == len(rag)
            assert n_inputs == len(feature_tasks)

            workflow_logger.info("LearnClassifierFromGt: call learning classifier for %i inputs" % n_inputs)
            for feat_tasks_i in feature_tasks:
                for feat in feat_tasks_i:
                    feat.open()
            self._learn_classifier_from_multiple_inputs(rag, gt, feature_tasks)
            for feat_tasks_i in feature_tasks:
                for feat in feat_tasks_i:
                    feat.close()

        else:
            workflow_logger.info("LearnClassifierFromGt: call learning classifier for single inputs")
            for feat in feature_tasks:
                feat.open()
            self._learn_classifier_from_single_input(rag, gt, feature_tasks)
            for feat in feature_tasks:
                feat.close()


    def _learn_classifier_from_single_input(self, rag, gt, feature_tasks):
        rag = rag.read()
        gt  = gt.read()

        # TODO correct for defects here
        nEdges = rag.numberOfEdges
        transitionEdge = rag.totalNumberOfInSliceEdges

        if self.learnXYOnly:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for xy edges only.")
            features, gt = self._learn_classifier_from_single_input_xy(rag, gt, feature_tasks, nEdges, transitionEdge)
        elif self.learnZOnly:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for z edges only.")
            features, gt = self._learn_classifier_from_single_input_z(rag, gt, feature_tasks, nEdges, transitionEdge)
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for all edges.")
            features = np.concatenate( [feat.read([0,0],feat.shape) for feat in feature_tasks], axis = 1 )

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = EdgeClassifier()
        classifier.train(features, gt)
        self.output().write(classifier.get_classifier())

    def _learn_classifier_from_single_input_xy(self, rag, gt, feature_tasks, nEdges, transitionEdge):
        gt = gt[:transitionEdge]
        features = []
        for feat_task in feature_tasks:
            assert feat_task.shape[0] == nEdges or feat_task[0] == transitionEdge
            if feat_task.shape[0] == nEdges:
                feat = feat_task.read([0,0],[transitionEdge,feat_task.shape[1]])
            else:
                feat = feat_task.read([0,0],feat_task.shape)
            features.append(feat)
        features = np.concatenate(features, axis = 1)
        return features, gt

    def _learn_classifier_from_single_input_z(self, rag, gt, feature_tasks, nEdges, transitionEdge):
        gt = gt[transitionEdge:]
        features = []
        for feat_task in feature_tasks:
            assert feat_task.shape[0] == nEdges or feat_task[0] == nEdges - transitionEdge
            if feat_task.shape[0] == nEdges:
                feat = feat_task.read([transitionEdge,0],[nEdges,feat_task.shape[1]])
            else:
                feat = feat_task.read([0,0],feat_task.shape)
            features.append(feat)
        features = np.concatenate(features, axis = 1)
        return features, gt


    def _learn_classifier_from_multiple_inputs(self, rag, gt, feature_tasks):

        if self.learnXYOnly:
            workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple input for xy edges only.")
            features, gt = self._learn_classifier_from_multiple_inputs_xy(rag, gt, feature_tasks)
        elif self.learnZOnly:
            workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple input for z edges only.")
            features, gt = self._learn_classifier_from_multiple_inputs_z(rag, gt, feature_tasks)
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple input for all edges.")
            features, gt = self._learn_classifier_from_multiple_inputs_all(rag, gt, feature_tasks)

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = EdgeClassifier()
        classifier.train(features, gt)
        self.output().write(classifier.get_classifier())


    def _learn_classifier_from_multiple_inputs_xy(self, rag, gt, feature_tasks):
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i].read()

            # TODO correct for defect pipeline here
            nEdges = rag_i.numberOfEdges
            transitionEdge = rag_i.totalNumberOfInSliceEdges

            gt_i = gt_i[:transitionEdge]

            features_i = []
            for feat_task in feat_tasks_i:
                assert feat_task.shape[0] == nEdges or feat_task.shape[0] == transitionEdge
                nFeats = feat_task.shape[1]
                feat = feat_task.read([0,0],[transitionEdge,nFeats])
                features_i.append(feat)
            features.append(np.concatenate(features_i, axis = 1))
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        return features, gts

    def _learn_classifier_from_multiple_inputs_z(self, rag, gt, feature_tasks):
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i].read()

            # TODO correct for defect pipeline here
            nEdges = rag_i.numberOfEdges
            transitionEdge = rag_i.totalNumberOfInSliceEdges

            gt_i = gt_i[transitionEdge:]

            features_i = []
            for feat_task in feat_tasks_i:
                assert feat_task.shape[0] == nEdges or feat_task.shape[0] == nEdges - transitionEdge
                nFeats = feat_task.shape[1]
                if feat_task.shape[0] == nEdges:
                    feat = feat_task.read([transitionEdge,0],[nEdges,nFeats])
                else:
                    feat = feat_task.read([0,0],[feat_task.shape[0],nFeats])
                features_i.append(feat)

            features.append(np.concatenate(features_i, axis = 1))
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        return features, gts

    def _learn_classifier_from_multiple_inputs_all(self, rag, gt, feature_tasks):
        features = []
        gts = []
        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            features.append(np.concatenate( [feat.read([0,0],feat.shape) for feat in feat_tasks_i], axis = 1 ))
            gts.append(gt[i].read())
        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        return features, gts


    def output(self):
        ninp_str = "SingleInput" if (len(self.pathsToSeg) == 1) else "MultipleInput"
        cf_str = "xgb" if PipelineParameter().useXGBoost else "sklearn"
        if self.learnXYOnly:
            edgetype_str = "xy"
        elif self.learnZOnly:
            edgetype_str = "z"
        else:
            edgetype_str = "all"
        save_path = os.path.join( PipelineParameter().cache,
            "LearnClassifierFromGt_%s_%s_%s.h5" % (ninp_str,cf_str,edgetype_str) )
        return PickleTarget(save_path)
