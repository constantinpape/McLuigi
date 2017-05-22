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

from concurrent import futures

import numpy as np
import vigra
import os

# import the proper nifty version
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty
    except ImportError:
        import nifty_with_gurobi as nifty
import nifty.graph.rag as nrag

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

# TODO random forest wrapper

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
        feature_tasks = inp["features"]
        for feat in feature_tasks:
            feat.openExisting()

        if PipelineParameter().defectPipeline:
            mod_adjacency = inp["modified_adjacency"]
            if mod_adjacency.read("has_defects"):
                nEdges = mod_adjacency.read("n_edges_modified")
                assert nEdges > 0, str(nEdges)
                workflow_logger.info("EdgeProbabilities: for defect corrected edges. Total number of edges: %i" % nEdges)
            else:
                nEdges = inp['rag'].readKey('numberOfEdges')
                workflow_logger.info("EdgeProbabilities: Total number of edges: %i" % nEdges)
        else:
            nEdges = inp['rag'].readKey('numberOfEdges')
            workflow_logger.info("EdgeProbabilities: Total number of edges: %i" % nEdges)

        out  = self.output()
        out.open([nEdges],[min(262144,nEdges)]) # 262144 = chunk size

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges separately.")
            self._predict_separate(feature_tasks, out)
        else:
            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges jointly out of core.")
                self._predict_joint_out_of_core(feature_tasks, out)
            else:
                workflow_logger.info("EdgeProbabilities: predicting xy - and z - edges jointly in core.")
                self._predict_joint_in_core(feature_tasks, out)

        for feat in feature_tasks:
            feat.close()
        out.close()


    def _predict_joint_in_core(self, feature_tasks, out):
        inp = self.input()
        classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_joined')

        features = np.concatenate(
                [feat.read([0,0], feat.shape(key), key) for key in ('features_xy', 'features_z') for feat in feature_tasks],
                axis = 1 )
        probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
        probsSub /= classifier.treeCount()
        probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
        out.write([0L],probs)

        # predict skip edges (for defects)
        if PipelineParameter().defectPipeline:
            features = np.concatenate( [feat.read([0,0], feat.shape('features_skip'), 'features_skip') for feat in feature_tasks], axis = 1 )
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_features_skip')
            probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probs /= classifier.treeCount()
            probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([feat_end],probs)


    # TODO change parallelisation to same logic as _predict_seperate_out_of_core
    def _predict_joint_out_of_core(self, feature_tasks, out):
        inp = self.input()
        classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_joined')
        nEdges = inp['rag'].readKey('numberOfEdges') - mod_adjacency.read('delete_edges').shape[0] if PipelineParameter().defectPipeline else inp['rag'].readKey('numberOfEdges')

        nEdgesXY = inp['rag'].readKey('totalNumberOfInSliceEdges')

        nSubFeats = PipelineParameter().nFeatureChunks
        assert nSubFeats > 1, nSubFeats
        featType = 'features_xy'
        changedType = False
        for subFeats in xrange(nSubFeats):
            print subFeats, '/', nSubFeats
            featIndexStart = int(float(subFeats)/nSubFeats * nEdges)
            featIndexEnd   = int(float(subFeats+1)/nSubFeats * nEdges)

            writeIndex = featIndexStart

            if featIndexEnd > nEdgesXY and featType == 'features_xy':
                featIndexEnd = nEdgesXY

            if featType == 'features_z':
                featIndexStart -= nEdgesXY
                featIndexEnd -= nEdgesXY

            if changedType:
                featIndexStart = 0
                writeIndex = nEdgesXY
                changedType = False

            if subFeats == nSubFeats:
                featIndexEnd = nEdges
            nEdgesSub = featIndexEnd - featIndexStart

            featuresSub = np.concatenate(
                    [feat.read([featIndexStart, 0], [featIndexEnd, feat.shape(featType)[1]], featType) for feat in feature_tasks],
                    axis = 1 )
            assert featuresSub.shape[0] == nEdgesSub
            probsSub = classifier.predictProbabilities(featuresSub.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probsSub /= classifier.treeCount()
            probsSub[np.isnan(probsSub)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([writeIndex],probsSub)

            if featIndexEnd == nEdgesXY:
                featType = 'features_z'
                changedType = True

        # predict skip edges (for defects)
        if PipelineParameter().defectPipeline:
            features = np.concatenate( [feat.read([0,0], feat.shape('features_skip'), 'features_skip') for feat in feature_tasks], axis = 1 )
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_defects')
            probs = classifier.predictProbabilities(features.astype('float32'), n_threads = PipelineParameter().nThreads)[:,1]
            probs /= classifier.treeCount()
            probs[np.isnan(probs)] = .5 # FIXME vigra rf3 produces nans some times
            out.write([feat_end],probs)


    def _predict_separate(self, feature_tasks, out):

        inp = self.input()
        if PipelineParameter().defectPipeline:
            nEdges = inp["modified_adjacency"].read("n_edges_modified")
            assert nEdges > 0
            nXYEdges = inp['rag'].readKey('totalNumberOfInSliceEdges')
            nZEdgesTotal = nEdges - nXYEdges
            nSkipEdges = inp["modified_adjacency"].read("skip_edges").shape[0]
        else:
            nEdges   = inp['rag'].readKey('numberOfEdges')
            nXYEdges = inp['rag'].readKey('totalNumberOfInSliceEdges')
            nZEdgesTotal  = nEdges - nXYEdges

        nFeatsXY = 0
        nFeatsZ  = 0
        for feat in feature_tasks:
            nFeatsXY += feat.shape('features_xy')[1]
            nFeatsZ  += feat.shape('features_z')[1]

        feat_types = ['features_xy', 'features_z', 'features_skip'] if PipelineParameter().defectPipeline else ['features_xy', 'features_z']
        for feat_type in feat_types:
            print "Predicting", feat_type, "features"
            classifier = vigra.learning.RandomForest3(str(self.pathToClassifier), 'rf_%s' % feat_type)

            if feat_type == 'features_xy':
                nEdgesType = nXYEdges
                nFeatsType = nFeatsXY
                startType  = 0
            elif feat_type == 'features_z':
                nEdgesType = nZEdgesTotal - nSkipEdges if PipelineParameter().defectPipeline else nZEdgesTotal
                nFeatsType = nFeatsZ
                startType  = nXYEdges
            elif feat_type == 'features_skip':
                nEdgesType = nSkipEdges
                nFeatsType = nFeatsZ
                startType  = nEdges - nSkipEdges

            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info("EdgeProbabilities: running separate prediction for %s edges out of core" % feat_type)
                self._predict_separate_out_of_core(classifier,
                        feature_tasks,
                        out,
                        feat_type,
                        nEdgesType,
                        nFeatsType,
                        startType)
            else:
                workflow_logger.info("EdgeProbabilities: running separate prediction for %s edges in core" % feat_type)
                self._predict_separate_in_core(classifier,
                        feature_tasks,
                        out,
                        feat_type,
                        nEdgesType,
                        nFeatsType,
                        startType)


    # In core prediction for edge type
    def _predict_separate_in_core(self,
            classifier,
            feature_tasks,
            out,
            featureType,
            nEdgesType,
            nFeatsType,
            startType):

            # In core prediction
            featuresType = np.zeros( (nEdgesType, nFeatsType), dtype = 'float32' )
            featOffset = 0

            for ii, feat in enumerate(feature_tasks):
                featuresType[:,featOffset:featOffset+feat.shape(featureType)[1]] = feat.read(
                    [0, 0], feat.shape(featureType), featureType)
                featOffset += feat.shape(featureType)[1]

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
            featureType,
            nEdgesType,
            nFeatsType,
            startType):

        nSubFeats = PipelineParameter().nFeatureChunks
        assert nSubFeats > 1, str(nSubFeats)

        def predict_subfeats(subFeatId):
            print subFeatId, '/', nSubFeats
            featIndexStart = int(float(subFeatId) / nSubFeats * nEdgesType)
            featIndexEnd   = int(float(subFeatId+1) / nSubFeats * nEdgesType)
            if subFeatId == nSubFeats:
                featIndexEnd = nEdgesType
            subFeats = np.concatenate(
                    [feat.read(
                        [featIndexStart,0],
                        [featIndexEnd, feat.shape(featureType)[1]],
                        featureType) for feat in feature_tasks ],
                    axis = 1)

            readStart = long(featIndexStart + startType)
            assert classifier.featureCount() == subFeats.shape[1], "%i , %i" % (classifier.featureCount(), subFeats.shape[1])
            probsSub = classifier.predictProbabilities(subFeats.astype('float32'), n_threads = 1)[:,1]
            probsSub /= classifier.treeCount()
            probsSub[np.isnan(probsSub)] = 0.5
            out.write([readStart], probsSub)
            return True

        nWorkers = PipelineParameter().nThreads
        #nWorkers = 1
        with futures.ThreadPoolExecutor(max_workers = nWorkers) as executor:
            tasks = []
            for subFeatId in xrange(nSubFeats):
                tasks.append( executor.submit( predict_subfeats, subFeatId ) )
            res = [t.result() for t in tasks]


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
        nodeGt = nrag.gridRagAccumulateLabels(rag, gt.get())
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
                    feat.openExisting()
            self._learn_classifier_from_multiple_inputs(rag, gt, feature_tasks)
            for feat_tasks_i in feature_tasks:
                for feat in feat_tasks_i:
                    feat.close()

        else:
            workflow_logger.info("LearnClassifierFromGt: call learning classifier for single inputs")
            for feat in feature_tasks:
                feat.openExisting()
            self._learn_classifier_from_single_input(rag, gt, feature_tasks)
            for feat in feature_tasks:
                feat.close()

    def _learn_classifier_from_single_input(self, rag, gt, feature_tasks):
        gt  = gt.read()
        inp = self.input()

        # correct for defects here
        if PipelineParameter().defectPipeline:
            # total number of edges
            nEdges = inp["modified_adjacency"][0].read("n_edges_modified")
            # starting index for z edges
            transitionEdge = rag.readKey('totalNumberOfInSliceEdges')
            # starting index for skip edges
            skipTransition = rag.readKey('numberOfEdges') - inp["modified_adjacency"][0].read("delete_edges").shape[0]
        else:
            nEdges = rag.readKey('numberOfEdges')
            transitionEdge = rag.readKey('totalNumberOfInSliceEdges')
            skipTransition = nEdges

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for xy and z edges separately.")
            self._learn_classifier_from_single_input_xy(gt, feature_tasks, transitionEdge)
            self._learn_classifier_from_single_input_z( gt, feature_tasks, transitionEdge, skipTransition)

        else:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for all edges.")
            features = np.concatenate(
                    [feat.read([0,0], feat.shape(key), key) for key in ('features_xy', 'features_z') for feat in feature_tasks],
                    axis = 1 )
            if PipelineParameter().defectPipeline:
                gt = gt[:skipTransition]
            assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
            classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                    treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
            classifier.writeHDF5( str(self.output().path), 'rf_joined' )

        if PipelineParameter().defectPipeline:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for skip edges (defects).")
            self._learn_classifier_from_single_input_defects(gt, feature_tasks, skipTransition)


    def _learn_classifier_from_single_input_xy(self, gt, feature_tasks, transitionEdge):
        gt = gt[:transitionEdge]
        features = []
        features = np.concatenate(
                [feat_task.read([0,0], feat_task.shape('features_xy'), 'features_xy') for feat_task in feature_tasks],
                axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_xy' )


    # if we learn with defects, we only consider the z edges that are not skip edges here
    def _learn_classifier_from_single_input_z(self, gt, feature_tasks, transitionEdge, skipTransition):
        gt = gt[transitionEdge:skipTransition] if PipelineParameter().defectPipeline else gt[transitionEdge:]
        features = np.concatenate(
                [feat_task.read([0,0], feat_task.shape('features_z'), 'features_z') for feat_task in feature_tasks],
                axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_z' )


    # if we learn with defects, we only consider the skip edges here
    def _learn_classifier_from_single_input_defects(self, gt, feature_tasks, skipTransition):
        assert PipelineParameter().defectPipeline
        gt = gt[skipTransition:]
        features = np.concatenate(
                [feat_task.read([0,0], feat_task.shape('features_skip'), 'features_skip') for feat_task in feature_tasks],
                axis = 1)
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gt.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_skip' )


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
            rag_i = rag[i]

            transitionEdge = rag_i.readKey('totalNumberOfInSliceEdges')
            gt_i = gt_i[:transitionEdge]
            features_i = np.concatenate(
                    [feat_task.read([0,0], feat_task.shape('features_xy'), 'features_xy') for feat_task in feat_tasks_i],
                    axis = 1)
            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_xy' )


    def _learn_classifier_from_multiple_inputs_z(self, rag, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for z called.")
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i]

            # if we learn with defects, we only keep the z edges that are not skip edges
            if PipelineParameter().defectPipeline:
                mod_i = self.input()["modified_adjacency"][i]
                if mod_i.read("has_defects"):
                    skipTransition = rag_i.readKey('numberOfEdges') - mod_i.read("delete_edges").shape[0]
                else:
                    skipTransition = rag_i.readKey('numberOfEdges')
            transitionEdge = rag_i.readKey('totalNumberOfInSliceEdges')

            gt_i = gt_i[transitionEdge:skipTransition] if PipelineParameter().defectPipeline else gt_i[transitionEdge:]
            features_i = np.concatenate(
                    [feat_task.read([0,0], feat_task.shape('features_z'), 'features_z') for feat_task in feat_tasks_i],
                    axis = 1)

            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_z' )


    def _learn_classifier_from_multiple_inputs_all(self, rag, gt, feature_tasks):
        features = []
        gts = []
        for i in xrange(len(gt)):

            rag_i = rag[i]
            if PipelineParameter().defectPipeline:
                if self.input()["modified_adjacency"][i].read("has_defects"):
                    skipTransition = rag_i.readKey('numberOfEdges') - self.input()["modified_adjacency"][i].read("delete_edges").shape[0]
                else:
                    skipTransition = rag_i.readKey('numberOfEdges')
            else:
                skipTransition = rag_i.readKey('numberOfEdges')

            features.append(
                    np.concatenate(
                        [feat.read([0,0], feat.shape(key), key) for key in ('features_xy', 'features_z') for feat in feature_tasks[i]],
                        axis = 1 ))
            gts.append(gt[i].read()[:skipTransition])

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
            rag_i = rag[i]

            # if we learn with defects, we only keep the z edges that are not skip edges
            mod_i = self.input()["modified_adjacency"][i]
            if not mod_i.read("has_defects"):
                continue

            nEdges = mod_i.read("n_edges_modified")
            skipTransition = rag_i.readKey('numberOfEdges') - mod_i.read("delete_edges").shape[0]
            transitionEdge = rag_i.readKey('totalNumberOfInSliceEdges')

            gt_i = gt_i[skipTransition:]
            features_i = np.concatenate(
                    [feat_task.read([0,0], feat_task.shape('features_skip'), 'features_skip') for feat_task in feat_tasks_i],
                    axis = 1)

            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate( features, axis = 0 )
        gts      = np.concatenate( gts )
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = vigra.learning.RandomForest3(features.astype('float32'), gts.astype('uint32'),
                treeCount = PipelineParameter().nTrees, max_depth = PipelineParameter().maxDepth, n_threads = PipelineParameter().nThreads)
        classifier.writeHDF5( str(self.output().path), 'rf_features_skip' )


    def output(self):
        ninp_str = "SingleInput" if (len(self.pathsToSeg) == 1) else "MultipleInput"
        save_path = os.path.join( PipelineParameter().cache, "LearnClassifierFromGt_%s.h5" % ninp_str )
        return HDF5DataTarget(save_path)
