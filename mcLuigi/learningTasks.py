# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from taskSelection import get_local_features,get_local_features_for_multiinp
from customTargets import PickleTarget, HDF5DataTarget, HDF5VolumeTarget
from dataTasks import DenseGroundtruth, ExternalSegmentation, StackedRegionAdjacencyGraph

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

# TODO log important params like n_rounds and n_trees
# wrapper for xgboost / sklearn.rf classifier
class EdgeClassifier(object):

    def __init__(self):
        self.use_xgb = PipelineParameter().useXGBoost
        self.cf = None

    def train(self, x, y):
        assert x.shape[0] == y.shape[0]
        if self.use_xgb:
            self._train_xgb(x,y)
        else:
            self._train_rf(x,y)

    def _train_xgb(self,x,y):
        import xgboost as xgb
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
        from sklearn.ensemble import RandomForestClassifier
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
        import xgboost as xgb
        xgb_mat = xgb.DMatrix(x)
        return self.cf.predict(x)[:,1]

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

    pathToSeg = luigi.Parameter()
    pathsToClassifier = luigi.ListParameter()

    def requires(self):

        if PipelineParameter().separateEdgeClassification:
            assert len(self.pathsToClassifier) == 2
        else:
            assert len(self.pathsToClassifier) == 1

        feature_tasks = get_local_features()

        return {"features" : feature_tasks,
                "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        # read the classifier parameter
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)
        nSubFeats = cf_config["nSubFeats"]

        # seperate classfiers for xy - and z - edges
        if PipelineParameter().separateEdgeClassification:

            rag = inp["rag"].read()
            features = inp["features"]

            nEdges   = rag.numberOfEdges
            nXYEdges = rag.totalNumberOfInSliceEdges
            nZEdges  = nEdges - nXYEdges

            nFeatsXY = 0
            nFeatsZ = 0
            for feat in features:
                feat.open()
                if feat.shape[0] == nEdges:
                    nFeatsXY += feat.shape[1]
                    nFeatsZ += feat.shape[1]
                elif feat.shape[0] == nXYEdges:
                    nFeatsXY += feat.shape[1]
                elif feat.shape[0] == nZEdges:
                    nFeatsZ += feat.shape[1]
                else:
                    raise RuntimeError("Number of features: " + str(feat.shape[0]) + " does not match number of edges (Total: " + str(nEdges) + ", XY: " + str(nXYEdges) + ", Z: " + str(nZEdges) + " )")

            out = self.output()

            out.open([nEdges],[min(262144,nEdges)])

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

                print "FeatType:", feat_type
                print "Edges from", startType, "to", endType

                # if nSubfeats > 1 we split feats to avoid memory error

                if nSubFeats > 1:

                    for subFeats in xrange(nSubFeats):
                        print subFeats, '/', nSubFeats
                        featIndexStart = int(float(subFeats)/nSubFeats * nEdgesType)
                        featIndexEnd   = int(float(subFeats+1)/nSubFeats * nEdgesType)
                        if subFeats == nSubFeats:
                            featIndexEnd = nEdgesType
                        nEdgesSub = featIndexEnd - featIndexStart

                        featuresType = np.zeros( (nEdgesSub, nFeatsTotal), dtype = 'float32' )
                        featOffset = 0

                        for ii, feat in enumerate(features):
                            #print "Loading feat", ii, "/", len(features)

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
                                raise RuntimeError("Number of features: " + str(nFeats) + " does not match number of edges (Total: " + str(nEdges) + ", XY: " + str(nXYEdges) + ", Z: " + str(nZEdges) + " )")

                            featOffset += feat.shape[1]

                        #featuresType = np.concatenate( featuresType, axis = 1 )
                        print "Features loaded, starting prediction"

                        readStart = long(featIndexStart + startType)
                        out.write([readStart], classifier.predict(featuresType))

                else: # we don't split the features along the edges

                    featuresType = np.zeros( (nEdgesType, nFeatsTotal), dtype = 'float32' )
                    featOffset = 0

                    for ii, feat in enumerate(features):
                        #print "Loading feat", ii, "/", len(features)

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
                            raise RuntimeError("Number of features: " + str(nFeats) + " does not match number of edges (Total: " + str(nEdges) + ", XY: " + str(nXYEdges) + ", Z: " + str(nZEdges) + " )")

                        featOffset += feat.shape[1]

                    print "Features loaded, starting prediction"
                    out.write([long(startType)], classifier.predict(featuresType))

            out.close()
            for feat in features:
                feat.close()

        # same classfier for xy - and z - edges
        else:

            feat_tasks = inp["features"]
            for feat in feat_tasks:
                feat.open()

            classifier = EdgeClassifier()
            classifier.load(self.pathsToClassifier[0])
            features = np.concatenate( [feat.read([0,0],feat.shape) for feat in inp["features"]], axis = 1 )

            for feat in feat_tasks:
                feat.close()

            probs = classifier.predict(features)

            out  = self.output()
            out.open([nEdges],[min(262144,nEdges)])
            self.output().write([0L],probs)

    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "EdgeProbabilities.h5"  )
        return HDF5VolumeTarget( save_path, 'float32'   )


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


class SingleClassifierFromGt(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()
    learnXYOnly  = luigi.BoolParameter(default = False)
    learnZOnly   = luigi.BoolParameter(default = False)

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        assert not (self.learnXYOnly and self.learnZOnly)
        feature_tasks = get_local_features(self.learnXYOnly, self.learnZOnly)
        return {"gt" : EdgeGroundtruth(self.pathToSeg, self.pathToGt),
                "features" : feature_tasks,
                "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        gt = inp["gt"].read()
        feature_tasks = inp["features"]

        if self.learnXYOnly:
            workflow_logger.info("SingleClassifierFromGt: learning classfier for xy edges.")

            rag = inp["rag"].read()
            nEdges = rag.numberOfEdges
            transitionEdge = rag.totalNumberOfInSliceEdges

            gt = gt[:transitionEdge]

            features = []
            for feat_task in feature_tasks:
                feat_task.open()
                assert feat_task.shape[0] == nEdges or feat_task[0] == transitionEdge
                if feat_task.shape[0] == nEdges:
                    feat = feat_task.read([0,0],[transitionEdge,feat_task.shape[1]])
                else:
                    feat = feat_task.read([0,0],feat_task.shape)
                features.append(feat)
                feat_task.close()
            features = np.concatenate(features, axis = 1)

        elif self.learnZOnly:
            workflow_logger.info("SingleClassifierFromGt: learning classfier for z edges.")

            rag = inp["rag"].read()
            nEdges = rag.numberOfEdges
            transitionEdge = rag.totalNumberOfInSliceEdges

            gt = gt[transitionEdge:]

            features = []
            for feat_task in feature_tasks:
                feat_task.open()
                assert feat_task.shape[0] == nEdges or feat_task[0] == nEdges - transitionEdge
                if feat_task.shape[0] == nEdges:
                    feat = feat_task.read([transitionEdge,0],[nEdges,feat_task.shape[1]])
                else:
                    feat = feat_task.read([0,0],feat_task.shape)
                features.append(feat)
                feat_task.close()
            features = np.concatenate(features, axis = 1)

        else:
            for feat_task in feature_tasks:
                feat_task.open()
            workflow_logger.info("SingleClassifierFromGt: learning classfier for all edges.")
            features = np.concatenate( [feat.read([0,0],feat.shape) for feat in feature_tasks], axis = 1 )
            for feat_task in feature_tasks:
                feat_task.close()

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        # read the classifier parameter
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)

        classifier = EdgeClassifier()
        classifier.train(features, gt)

        self.output().write(classifier.get_classifier())


    def output(self):

        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)

        save_path = os.path.join( PipelineParameter().cache, "SingleClassifierFromGt"  )

        if PipelineParameter().useXGBoost:
            save_path += "_xgb"
        else:
            save_path += "_sklearn"

        if self.learnXYOnly:
            save_path += "_xy"
        if self.learnZOnly:
            save_path += "_z"
        save_path += ".pkl"

        return PickleTarget(save_path)


class SingleClassifierFromMultipleInputs(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToGt  = luigi.ListParameter()
    learnXYOnly  = luigi.BoolParameter(default = False)
    learnZOnly   = luigi.BoolParameter(default = False)

    def requires(self):
        # This way of generating the features is quite hacky, but it is the
        # least ugly way I could come up with till now.
        # and as long as we only use the pipeline for deploymeny it should be alright
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        feature_tasks = get_local_features_for_multiinp(self.learnXYOnly, self.learnZOnly)
        assert len(feature_tasks) == len(self.pathsToSeg)
        gts = [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in xrange(len(self.pathsToSeg))]
        rag_tasks = [StackedRegionAdjacencyGraph(segp) for segp in self.pathsToSeg]
        return {"gts" : gts, "features" : feature_tasks, "rags" : rag_tasks}

    @run_decorator
    def run(self):

        inp = self.input()

        gt_tasks = inp["gts"]
        feature_tasks = inp["features"]

        rag_tasks = inp["rags"]

        assert len(feature_tasks) == len(gt_tasks)

        features = []
        gt = []

        for i in xrange(len(gt_tasks)):

            feat_tasks_i = feature_tasks[i]
            gt_i = gt_tasks[i].read()

            if self.learnXYOnly:

                rag = rag_tasks[i].read()
                nEdges = rag.numberOfEdges
                transitionEdge = rag.totalNumberOfInSliceEdges

                gt_i = gt_i[:transitionEdge]

                features_i = []
                for feat_task in feat_tasks_i:
                    feat_task.open()
                    assert feat_task.shape[0] == nEdges or feat_task.shape[0] == transitionEdge
                    nFeats = feat_task.shape[1]
                    feat = feat_task.read([0,0],[transitionEdge,nFeats])
                    features_i.append(feat)
                    feat_task.close()
                features_i = np.concatenate(features_i, axis = 1)

            elif self.learnZOnly:

                rag = rag_tasks[i].read()
                nEdges = rag.numberOfEdges
                transitionEdge = rag.totalNumberOfInSliceEdges

                gt_i = gt_i[transitionEdge:]

                features_i = []
                for feat_task in feat_tasks_i:
                    feat_task.open()
                    assert feat_task.shape[0] == nEdges or feat_task.shape[0] == nEdges - transitionEdge
                    nFeats = feat_task.shape[1]
                    if feat_task.shape[0] == nEdges:
                        feat = feat_task.read([transitionEdge,0],[nEdges,nFeats])
                    else:
                        feat = feat_task.read([0,0],[feat_task.shape[0],nFeats])
                    features_i.append(feat)
                    feat_task.close()
                features_i = np.concatenate(features_i, axis = 1)

            else:
                for feat_task in feat_tasks_i:
                    feat_task.open()
                workflow_logger.info("SingleClassifierFromMultipleInputs: learning classfier for all edges.")
                features_i = np.concatenate( [feat.read([0,0],feat.shape) for feat in feat_tasks_i], axis = 1 )

            assert features_i.shape[0] == gt_i.shape[0]

            features.append(features_i)
            gt.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gt    = np.concatenate( gt )

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])

        # read the classifier parameter
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)

        classifier = EdgeClassifier()
        classifier.train(features, gt)
        self.output().write(classifier.get_classifier())

    def output(self):
        with open(PipelineParameter().EdgeClassifierConfigFile, 'r') as f:
            cf_config = json.load(f)

        save_path = os.path.join( PipelineParameter().cache, "SingleClassifierFromMultipleInputs"  )

        if PipelineParameter().useXGBoost:
            save_path += "_xgb"
        else:
            save_path += "_sklearn"

        if self.learnXYOnly:
            save_path += "_xy"
        if self.learnZOnly:
            save_path += "_z"
        save_path += ".pkl"

        return PickleTarget(save_path)
