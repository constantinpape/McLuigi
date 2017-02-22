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
import vigra
import nifty
import os
import time
import json
import cPickle as pickle

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class EdgeProbabilities(luigi.Task):

    pathToSeg        = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    def requires(self):
        return_tasks ={"features" : get_local_features(), "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return return_tasks

    @run_decorator
    def run(self):
        assert os.path.exists(self.pathToClassifier), self.pathToClassifier
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
        classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_joined')
        if PipelineParameter.defectPipeline():
            mod_adjacency = self.input()['modified_adjacency']
            feat_end = rag.numberOfEdges - mod_adjacency.read('delete_edges').shape[0]
            n_edges = mod_adjacency.read('n_edges_modified')
        else:
            feat_end = rag.numberOfEdges

        features = np.concatenate( [feat.read([0,0],[feat_end,feat.shape[1]]) for feat in feature_tasks], axis = 1 )
        probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
        probsSub /= classifier.treeCount()
        probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
        out.write([0L],probs)

        # predict skip edges (for defects)
        if PipelineParameter().defectPipeline:
            features = np.concatenate( [feat.read([feat_end,0],[n_edges,feat.shape[1]]) for feat in feature_tasks], axis = 1 )
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_defects')
            probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probs /= classifier.treeCount()
            probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([feat_end],probs)


    def _predict_joint_out_of_core(self, rag, feature_tasks, out):
        classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_joined')
        nEdges = rag.numberOfEdges - mod_adjacency.read('delete_edges').shape[0] if PipelineParameter().defectPipeline else rag.numberOfEdges

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
            probsSub = classifier.predictProbabilities(featuresSub.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probsSub /= classifier.treeCount()
            probsSub[np.isnan(probsSub)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([featIndexStart],probsSub)

        # predict skip edges (for defects)
        if PipelineParameter().defectPipeline:
            features = np.concatenate( [feat.read([nEdges,0], feat.shape) for feat in feature_tasks], axis = 1 )
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_defects')
            probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probs /= classifier.treeCount()
            probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([feat_end],probs)


    def _predict_separate(self, rag, feature_tasks, out):

        inp = self.input()
        if PipelineParameter().defectPipeline:
            nEdges = inp["modified_adjacency"].read("n_edges_modified")
            assert nEdges > 0
            nXYEdges = rag.totalNumberOfInSliceEdges
            nZEdgesTotal = nEdges - nXYEdges
            nSkipEdges = inp["modified_adjacency"].read("skip_edges").shape[0]
        else:
            nEdges   = rag.numberOfEdges
            nXYEdges = rag.totalNumberOfInSliceEdges
            nZEdgesTotal  = nEdges - nXYEdges

        nFeatsXY = 0
        nFeatsZ  = 0
        for feat in feature_tasks:
            if feat.shape[0] == nEdges:
                nFeatsXY += feat.shape[1]
                nFeatsZ += feat.shape[1]
            elif feat.shape[0] == nXYEdges:
                nFeatsXY += feat.shape[1]
            elif feat.shape[0] == nZEdgesTotal:
                nFeatsZ += feat.shape[1]
            else:
                raise RuntimeError("Number of features: %i does not match number of edges (Total: %i, XY: %i, Z: %i )" % (feat.shape[0],nEdges,nXYEdges,nZEdges))

        feat_types = ['xy', 'z', 'defects'] if PipelineParameter().defectPipeline else ['xy', 'z']
        for feat_type in feat_types:
            print "Predicting", feat_type, "features"
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_%s' % feat_type)

            if feat_type == 'xy':
                nEdgesType = nXYEdges
                nEdgesTypeTotal = nXYEdges
                nEdgesNonType = nZEdgesTotal
                startType  = 0
                endType    = nXYEdges
                nFeatsTotal = nFeatsXY
            elif feat_type == 'z':
                nEdgesType = nZEdgesTotal - nSkipEdges if PipelineParameter().defectPipeline else nZEdgesTotal
                nEdgesTypeTotal = nZEdgesTotal
                nEdgesNonType = nXYEdges
                startType = nXYEdges
                endType = nEdges - nSkipEdges if PipelineParameter().defectPipeline else nEdges
                nFeatsTotal = nFeatsZ
            elif feat_type == 'defects':
                nEdgesType = nSkipEdges
                nEdgesTypeTotal = nZEdgesTotal
                nEdgesNonType = nXYEdges
                startType = nEdges - nSkipEdges
                endType = nEdges
                nFeatsTotal = nFeatsZ

            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info("EdgeProbabilities: running separate prediction for %s edges out of core" % feat_type)
                self._predict_separate_out_of_core(classifier,
                        feature_tasks,
                        out,
                        nEdges,
                        nEdgesType,
                        nEdgesTypeTotal,
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
                        nEdgesTypeTotal,
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
            nEdgesTypeTotal,
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
                elif nFeats == nEdgesTypeTotal:
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
            probs = classifier.predictProbabilities(featuresType.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probs /= classifier.treeCount()
            probs[np.isnan(probs)] = 0.5
            out.write([long(startType)], probs)


    # Out of core prediction for edge type
    def _predict_separate_out_of_core(self,
            classifier,
            feature_tasks,
            out,
            nEdges,
            nEdgesType,
            nEdgesTypeTotal,
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
                elif nFeats == nEdgesTotalType:
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

            print "Features loaded, starting prediction"
            readStart = long(featIndexStart + startType)
            assert classifier.featureCount() == featuresType.shape[1], "%i , %i" % (classifier.featureCount(), featuresType.shape[1])
            probsSub = classifier.predictProbabilities(featuresType.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probsSub /= classifier.treeCount()
            probsSub[np.isnan(probsSub)] = 0.5
            out.write([readStart], probsSub)

    def output(self):
        save_path = os.path.join(PipelineParameter().cache, "EdgeProbabilities%s_%s.h5" % ("Separate" if PipelineParameter().separateEdgeClassification else "Joint",
            "modified" if PipelineParameter().defectPipeline else "standard"))
        return HDF5VolumeTarget(save_path, 'float32')


# TODO fuzzy mapping in nifty ?!
class EdgeGroundtruth(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        if PipelineParameter().defectPipeline:
            return {"gt" : DenseGroundtruth(self.pathToGt),
                    "rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                    "modified_adjacency" : ModifiedAdjacency(self.pathToSeg)}
        else:
            return {"gt" : DenseGroundtruth(self.pathToGt), "rag" : StackedRegionAdjacencyGraph(self.pathToSeg) }

    @run_decorator
    def run(self):
        inp = self.input()
        gt = inp["gt"]
        gt.open()
        rag = inp["rag"].read()
        nodeGt = nifty.graph.rag.gridRagAccumulateLabels(rag, gt.get())
        if PipelineParameter().defectPipeline:
            mod_adjacency = nifty.graph.UndirectedGraph()
            if inp["modified_adjacency"].read("has_defects"):
                mod_adjacency.deserialize(inp["modified_adjacency"].read("modified_adjacency"))
                uvIds = mod_adjacency.uvIds()
            else:
                uvIds = rag.uvIds()
        else:
            uvIds = rag.uvIds()
        uGt = nodeGt[ uvIds[:,0] ]
        vGt = nodeGt[ uvIds[:,1] ]
        edgeGt = (uGt != vGt).astype(np.uint8)
        assert (np.unique(edgeGt) == np.array([0,1])).all(), str(np.unique(edgeGt))
        assert edgeGt.shape[0] == uvIds.shape[0]
        self.output().write(edgeGt)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        def_str = 'modified' if PipelineParameter().defectPipeline else 'standard'
        save_path = os.path.join( PipelineParameter().cache, "EdgeGroundtruth_%s_%s.h5" % (segFile,def_str) )
        return HDF5DataTarget( save_path  )


class LearnClassifierFromGt(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToGt  = luigi.ListParameter()

    def requires(self):
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        n_inputs = len(self.pathsToSeg)

        if n_inputs == 1:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from single input")
            feature_tasks = get_local_features()
            return_tasks = {"gt" : EdgeGroundtruth(self.pathsToSeg[0], self.pathsToGt[0]),
                            "features" : feature_tasks,
                            "rag" : StackedRegionAdjacencyGraph(self.pathsToSeg[0])}
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from %i inputs" % n_inputs)
            feature_tasks = get_local_features_for_multiinp()
            assert len(feature_tasks) == n_inputs
            return_tasks = {"gt" : [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in xrange(n_inputs)],
                            "features" : feature_tasks,
                            "rag" : [StackedRegionAdjacencyGraph(segp) for segp in self.pathsToSeg]}
        if PipelineParameter().defectPipeline:
            return_tasks['modified_adjacency'] = [ModifiedAdjacency(self.pathsToSeg[i]) for i in xrange(n_inputs)]
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
        inp = self.input()

        # correct for defects here
        if PipelineParameter().defectPipeline:
            # total number of edges
            nEdges = inp["modified_adjacency"][0].read("n_edges_modified")
            # starting index for z edges
            transitionEdge = rag.totalNumberOfInSliceEdges
            # starting index for skip edges
            skipTransition = rag.numberOfEdges - inp["modified_adjacency"][0].read("delete_edges").shape[0]
        else:
            nEdges = rag.numberOfEdges
            transitionEdge = rag.totalNumberOfInSliceEdges
            skipTransition = nEdges

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for xy and z edges separately.")
            self._learn_classifier_from_single_input_xy(gt, feature_tasks, nEdges, transitionEdge)
            self._learn_classifier_from_single_input_z( gt, feature_tasks, nEdges, transitionEdge, skipTransition)

        else:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for all edges.")
            feat_end = skipTransition if PipelineParameter().defectPipeline else nEdges
            features = np.concatenate( [feat.read([0,0],[feat_end,feat.shape[1]]) for feat in feature_tasks], axis = 1 )
            if PipelineParameter().defectPipeline:
                gt = gt[:skipTransition]
            assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
            classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                    treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
            classifier.writeHDF5( str(self.output().path), 'rf_joined' )

        if PipelineParameter().defectPipeline:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for skip edges (defects).")
            self._learn_classifier_from_single_input_defects(gt, feature_tasks, nEdges, transitionEdge, skipTransition)


    def _learn_classifier_from_single_input_xy(self, gt, feature_tasks, nEdges, transitionEdge):
        gt = gt[:transitionEdge]
        features = []
        for feat_task in feature_tasks:
            assert feat_task.shape[0] in (nEdges , transitionEdge, nEdges - transitionEdge)
            if feat_task.shape[0] == nEdges:
                feat = feat_task.read([0,0],[transitionEdge,feat_task.shape[1]])
            elif feat_task.shape[0] == transitionEdge:
                feat = feat_task.read([0,0],feat_task.shape)
            else:
                continue
            features.append(feat)
        features = np.concatenate(features, axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_xy' )


    # if we learn with defects, we only consider the z edges that are not skip edges here
    def _learn_classifier_from_single_input_z(self, gt, feature_tasks, nEdges, transitionEdge, skipTransition):
        gt = gt[transitionEdge:skipTransition] if PipelineParameter().defectPipeline else gt[transitionEdge:]
        features = []
        for feat_task in feature_tasks:
            assert feat_task.shape[0] in (nEdges, nEdges - transitionEdge, transitionEdge)
            if feat_task.shape[0] == nEdges:
                feat = feat_task.read([transitionEdge,0],
                    [skipTransition if PipelineParameter().defectPipeline else nEdges, feat_task.shape[1]])
            elif feat_task.shape[0] == nEdges - transitionEdge:
                feat = feat_task.read([0,0],
                    [skipTransition - transitionEdge  if PipelineParameter().defectPipeline else feat_task.shape[0],feat_task.shape[1]])
            else:
                continue
            features.append(feat)
        features = np.concatenate(features, axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_z' )


    # if we learn with defects, we only consider the skip edges here
    def _learn_classifier_from_single_input_defects(self, gt, feature_tasks, nEdges, transitionEdge, skipTransition):
        assert PipelineParameter().defectPipeline
        gt = gt[skipTransition:]
        features = []
        for feat_task in feature_tasks:
            assert feat_task.shape[0] in (nEdges, nEdges - transitionEdge, transitionEdge)
            if feat_task.shape[0] == nEdges:
                feat = feat_task.read([skipTransition,0],[nEdges,feat_task.shape[1]])
            elif feat_task.shape[0] == nEdges -transitionEdge:
                feat = feat_task.read([0,0],[skipTransition - transitionEdge, feat_task.shape[1]])
            else:
                continue
            features.append(feat)
        features = np.concatenate(features, axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_defects' )


    def _learn_classifier_from_multiple_inputs(self, rag, gt, feature_tasks):

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for xy and z edges separately.")
            self._learn_classifier_from_multiple_inputs_xy(rag, gt, feature_tasks)
            self._learn_classifier_from_multiple_inputs_z(rag, gt, feature_tasks)

        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for all edges jointly.")
            self._learn_classifier_from_multiple_inputs_all(rag, gt, feature_tasks)

        if PipelineParameter().defectPipeline:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from multiple inputs for skip edges (defects).")
            self._learn_classifier_from_multiple_inputs_defects(rag, gt, feature_tasks)


    def _learn_classifier_from_multiple_inputs_xy(self, rag, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for xy called.")
        features = []
        gts = []

        inp = self.input()
        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i].read()

            # correct for defect pipeline here
            if PipelineParameter().defectPipeline:
                if inp["modified_adjacency"][i].read("has_defects"):
                    nEdges = inp["modified_adjacency"][i].read("n_edges_modified")
                else:
                    nEdges = rag_i.numberOfEdges
            else:
                nEdges = rag_i.numberOfEdges
            transitionEdge = rag_i.totalNumberOfInSliceEdges

            gt_i = gt_i[:transitionEdge]

            features_i = []
            for feat_task in feat_tasks_i:
                assert feat_task.shape[0] in (nEdges, transitionEdge, nEdges - transitionEdge)
                nFeats = feat_task.shape[1]
                if feat_task.shape[0] in (nEdges, transitionEdge):
                    feat = feat_task.read([0,0],[transitionEdge,nFeats])
                else:
                    continue
                features_i.append(feat)
            features.append(np.concatenate(features_i, axis = 1))
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_xy' )


    def _learn_classifier_from_multiple_inputs_z(self, rag, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for z called.")
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i].read()

            # if we learn with defects, we only keep the z edges that are not skip edges
            if PipelineParameter().defectPipeline:
                mod_i = self.input()["modified_adjacency"][i]
                if mod_i.read("has_defects"):
                    nEdges = mod_i.read("n_edges_modified")
                    skipTransition = rag_i.numberOfEdges - mod_i.read("delete_edges").shape[0]
                else:
                    nEdges = rag_i.numberOfEdges
                    skipTransition = rag_i.numberOfEdges
            else:
                nEdges = rag_i.numberOfEdges
            transitionEdge = rag_i.totalNumberOfInSliceEdges

            gt_i = gt_i[transitionEdge:skipTransition] if PipelineParameter().defectPipeline else gt_i[transitionEdge:]

            features_i = []
            for feat_task in feat_tasks_i:
                assert feat_task.shape[0] in (nEdges, transitionEdge, nEdges - transitionEdge)
                nFeats = feat_task.shape[1]
                if feat_task.shape[0] == nEdges:
                    feat = feat_task.read([transitionEdge,0],
                        [skipTransition if PipelineParameter().defectPipeline else nEdges, nFeats])
                elif feat_task.shape[0] == nEdges - transitionEdge:
                    feat = feat_task.read([0,0],
                        [skipTransition - transitionEdge if PipelineParameter().defectPipeline else feat_task.shape[0], nFeats])
                else:
                    continue
                features_i.append(feat)

            features.append(np.concatenate(features_i, axis = 1))
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_z' )


    def _learn_classifier_from_multiple_inputs_all(self, rag, gt, feature_tasks):
        features = []
        gts = []
        for i in xrange(len(gt)):

            if PipelineParameter().defectPipeline:
                if self.input()["modified_adjacency"][i].read("has_defects"):
                    skipTransition = rag[i].numberOfEdges - self.input()["modified_adjacency"][i].read("delete_edges").shape[0]
                else:
                    skipTransition = rag_i.numberOfEdges

            feat_tasks_i = feature_tasks[i]
            feat_end = skipTransition if PipelineParameter().defectPipeline else rag.numberOfEdges
            features.append(np.concatenate( [feat.read([0,0], [feat_end,feat.shape[1]]) for feat in feat_tasks_i], axis = 1 ))
            gts.append(gt[i].read())

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_joined' )


    def _learn_classifier_from_multiple_inputs_defects(self, rag, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for defects called.")
        assert PipelineParameter().defectPipeline
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i].read()

            # if we learn with defects, we only keep the z edges that are not skip edges
            mod_i = self.input()["modified_adjacency"][i]
            if not mod_i.read("has_defects"):
                continue

            nEdges = mod_i.read("n_edges_modified")
            skipTransition = rag_i.numberOfEdges - mod_i.read("delete_edges").shape[0]
            transitionEdge = rag_i.totalNumberOfInSliceEdges

            gt_i = gt_i[skipTransition:]

            features_i = []
            for feat_task in feat_tasks_i:
                assert feat_task.shape[0] in (nEdges, transitionEdge, nEdges - transitionEdge)
                nFeats = feat_task.shape[1]
                if feat_task.shape[0] == nEdges:
                    feat = feat_task.read([skipTransition,0], [nEdges,nFeats])
                elif feat_task.shape[0] == nEdges - transitionEdge:
                    feat = feat_task.read([0,0], [skipTransition - transitionEdge, nFeats])
                else:
                    continue
                features_i.append(feat)

            features.append(np.concatenate(features_i, axis = 1))
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_defects' )


    def output(self):
        ninp_str = "SingleInput" if (len(self.pathsToSeg) == 1) else "MultipleInput"
        save_path = os.path.join( PipelineParameter().cache, "LearnClassifierFromGt_%s.h5" % ninp_str )
        return HDF5DataTarget(save_path)
