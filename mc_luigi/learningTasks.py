# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from taskSelection import get_local_features, get_local_features_for_multiinp
from customTargets import HDF5DataTarget, HDF5VolumeTarget, FolderTarget
from dataTasks import DenseGroundtruth, StackedRegionAdjacencyGraph
from defectHandlingTasks import ModifiedAdjacency

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging

from concurrent import futures

import numpy as np
import vigra
import os
import h5py

# import the proper nifty version
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi as nifty
        import nifty_with_gurobi.graph.rag as nrag

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

try:
    from sklearn.ensemble import RandomForestClassifier as RFType
    use_sklearn = True
    import cPickle as pickle
    workflow_logger.info("Using sklearn random forest")
except ImportError:
    RFType = vigra.learning.RandomForest3
    use_sklearn = False
    workflow_logger.info("Using vigra random forest 3")


# wrapper for sklearn / random forest
class RandomForest(object):

    def __init__(
            self,
            train_data,
            train_labels,
            n_trees,
            n_threads,
            max_depth=None
    ):

        if isinstance(train_data, str) and train_data == '__will_deserialize__':
            return
        else:
            assert isinstance(train_data, np.ndarray)

        assert train_data.shape[0] == train_labels.shape[0]
        self.n_threads = n_threads
        self.n_trees = n_trees
        self.max_depth = max_depth

        if use_sklearn:
            self._learn_rf_sklearn(train_data, train_labels)
        else:
            self._learn_rf_vigra(train_data, train_labels)

    @classmethod
    def load_from_file(self, file_path, key):
        assert os.path.exists(file_path), file_path
        self = self('__will_deserialize__', None, None, None)
        if use_sklearn:
            save_path = os.path.join(file_path, "%s.pkl" % key)
            assert os.path.exists(save_path)
            with open(save_path) as f:
                rf = pickle.load(f)
            self.n_trees = rf.n_estimators
        else:
            save_path = file_path + ".h5"
            rf = RFType(save_path, key)
            self.n_trees = rf.treeCount()
        self.rf = rf
        return self

    @staticmethod
    def has_defect_rf(file_path):
        if use_sklearn:
            return os.path.exists(os.path.join(file_path, "rf_defects.pkl"))
        else:
            with h5py.File(file_path + '.h5') as f:
                return 'rf_defects' in f.keys()

    @classmethod
    def is_cached(self, file_path):
        save_path = file_path if use_sklearn else file_path + ".h5"
        return os.path.exists(save_path)

    def _learn_rf_sklearn(self, train_data, train_labels):
        self.rf = RFType(
            n_estimators=self.n_trees,
            n_jobs=self.n_threads,
            verbose=2,
            max_depth=self.max_depth
        )
        self.rf.fit(train_data, train_labels)

    def _learn_rf_vigra(self, train_data, train_labels):
        self.rf = RFType(
            train_data,
            train_labels,
            treeCount=self.n_trees,
            n_threads=self.n_threads,
            max_depth=self.max_depth if self.max_depth is not None else 0
        )

    def predict_probabilities(self, test_data, n_threads=1):
        if use_sklearn:
            return self._predict_sklearn(test_data, n_threads)
        else:
            return self._predict_vigra(test_data, n_threads)

    def _predict_sklearn(self, test_data, n_threads):
        if self.rf.n_jobs != n_threads:
            self.rf.n_jobs = n_threads
        return self.rf.predict_proba(test_data)

    def _predict_vigra(self, test_data, n_threads):
        prediction = self.rf.predictProbabilities(test_data, n_threads=n_threads)
        # normalize the prediction
        prediction /= self.n_trees
        # normalize by the number of trees and remove nans
        prediction[np.isnan(prediction)] = .5
        prediction[np.isinf(prediction)] = .5
        assert prediction.max() <= 1.
        return prediction

    def write(self, file_path, key):
        if use_sklearn:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            save_path = os.path.join(file_path, "%s.pkl" % (key))
            with open(save_path, 'w') as f:
                pickle.dump(self.rf, f)
        else:
            save_path = file_path + ".h5"
            self.rf.writeHDF5(file_path, key)


class EdgeProbabilities(luigi.Task):

    pathToSeg        = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    def requires(self):
        return_tasks = {
            "features": get_local_features(),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg)
        }
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
                n_edges = mod_adjacency.read("n_edges_modified")
                assert n_edges > 0, str(n_edges)
                workflow_logger.info(
                    "EdgeProbabilities: for defect corrected edges. Total number of edges: %i" % n_edges
                )

            else:
                n_edges = inp['rag'].readKey('numberOfEdges')
                workflow_logger.info("EdgeProbabilities: Total number of edges: %i" % n_edges)

        else:
            n_edges = inp['rag'].readKey('numberOfEdges')
            workflow_logger.info("EdgeProbabilities: Total number of edges: %i" % n_edges)

        out  = self.output()
        out.open([n_edges], [min(262144, n_edges)])  # 262144 = chunk size

        self._predict(feature_tasks, out)

        for feat in feature_tasks:
            feat.close()
        out.close()

    def _predict(self, feature_tasks, out):
        inp = self.input()

        # check if we predict for a dataset with defects
        # we need to do this so clumsily, because 'modified_adjacency' does not exist
        # as a key in inp if defectPipeline == False
        # TODO this could be done better with a get("modified_adjacency") and a default value
        has_defects = False
        if PipelineParameter().defectPipeline:
            if inp["modified_adjacency"].read("has_defects"):
                has_defects = True

        if has_defects:
            n_edges = inp["modified_adjacency"].read("n_edges_modified")
            assert n_edges > 0
            n_edges_xy = inp['rag'].readKey('totalNumberOfInSliceEdges')
            n_edges_total_z = n_edges - n_edges_xy
            n_edges_skip = inp["modified_adjacency"].read("skip_edges").shape[0]
        else:
            n_edges   = inp['rag'].readKey('numberOfEdges')
            n_edges_xy = inp['rag'].readKey('totalNumberOfInSliceEdges')
            n_edges_total_z  = n_edges - n_edges_xy

        n_feats_xy = 0
        n_feats_z  = 0
        n_feats_skip = 0
        for feat in feature_tasks:
            n_feats_xy += feat.shape('features_xy')[1]
            n_feats_z  += feat.shape('features_z')[1]
            if has_defects:
                n_feats_skip += feat.shape('features_skip')[1]

        feat_types = ['features_xy', 'features_z']
        classifier_types = feat_types if PipelineParameter().separateEdgeClassification else 2 * ['features_joined']
        if has_defects:
            feat_types += ['features_skip']
            classifier_types += ['features_skip']

        for ii, feat_type in enumerate(feat_types):
            workflow_logger.info("Predicting features for %s" % feat_type)
            classifier = RandomForest.load_from_file(
                str(self.pathToClassifier),
                "rf_%s" % classifier_types[ii]
            )

            if feat_type == 'features_xy':
                n_edges_type = n_edges_xy
                n_feats_type = n_feats_xy
                start_type  = 0
            elif feat_type == 'features_z':
                n_edges_type = n_edges_total_z - n_edges_skip if has_defects else n_edges_total_z
                n_feats_type = n_feats_z
                start_type  = n_edges_xy
            elif feat_type == 'features_skip':
                n_edges_type = n_edges_skip
                n_feats_type = n_feats_skip
                start_type  = n_edges - n_edges_skip

            if PipelineParameter().nFeatureChunks > 1:
                workflow_logger.info(
                    "EdgeProbabilities: predicting %s edges with seperate claissifier out of core." % feat_type
                    if PipelineParameter().separateEdgeClassification else
                    "EdgeProbabilities: predicting %s edges with joint classifier out of core" % feat_type
                )
                self._predict_out_of_core(
                    classifier,
                    feature_tasks,
                    out,
                    feat_type,
                    n_edges_type,
                    n_feats_type,
                    start_type
                )
            else:
                workflow_logger.info(
                    "EdgeProbabilities: predicting %s edges with seperate claissifier in core." % feat_type
                    if PipelineParameter().separateEdgeClassification else
                    "EdgeProbabilities: predicting %s edges with joint classifier in core" % feat_type
                )
                self._predict_in_core(
                    classifier,
                    feature_tasks,
                    out,
                    feat_type,
                    n_edges_type,
                    n_feats_type,
                    start_type
                )

    # In core prediction for edge type
    def _predict_in_core(
            self,
            classifier,
            feature_tasks,
            out,
            feat_type,
            n_edges,
            n_feats,
            start
    ):
        # In core prediction
        features = np.zeros((n_edges, n_feats), dtype='float32')
        offset = 0

        for ii, feat in enumerate(feature_tasks):
            features[:, offset:offset + feat.shape(feat_type)[1]] = feat.read(
                [0, 0], feat.shape(feat_type), feat_type)
            offset += feat.shape(feat_type)[1]

        probs = classifier.predict_probabilities(features, PipelineParameter().nThreads)[:, 1]
        out.write([long(start)], probs)

    # Out of core prediction for edge type
    def _predict_out_of_core(
            self,
            classifier,
            feature_tasks,
            out,
            feat_type,
            n_edges,
            n_feats,
            start
    ):
        n_sub_feats = PipelineParameter().nFeatureChunks
        assert n_sub_feats > 1, str(n_sub_feats)

        def predict_subfeats(sub_feat_id):
            print sub_feat_id, '/', n_sub_feats
            start_index = int(float(sub_feat_id) / n_sub_feats * n_edges)
            end_index   = int(float(sub_feat_id + 1) / n_sub_feats * n_edges)
            if sub_feat_id == n_sub_feats:
                end_index = n_edges
            sub_feats = np.concatenate(
                [feat.read(
                    [start_index, 0],
                    [end_index, feat.shape(feat_type)[1]],
                    feat_type) for feat in feature_tasks],
                axis=1
            )

            read_start = long(start_index + start)

            probs = classifier.predict_probabilities(sub_feats, 1)[:, 1]
            out.write([read_start], probs)
            return True

        n_workers = PipelineParameter().nThreads
        # n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(predict_subfeats, sub_feat_id) for sub_feat_id in xrange(n_sub_feats)]
            [t.result() for t in tasks]

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "EdgeProbabilities%s_%s.h5" % (
                "Separate" if PipelineParameter().separateEdgeClassification else "Joint",
                "modified" if PipelineParameter().defectPipeline else "standard"
            )
        )
        return HDF5VolumeTarget(save_path, 'float32')


# TODO fuzzy mapping in nifty ?!
# -> we can use nifty.ground_truth for fuzzy gt
class EdgeGroundtruth(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToGt  = luigi.Parameter()

    def requires(self):
        if PipelineParameter().defectPipeline:
            return {
                "gt": DenseGroundtruth(self.pathToGt),
                "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
                "modified_adjacency": ModifiedAdjacency(self.pathToSeg)
            }
        else:
            return {"gt": DenseGroundtruth(self.pathToGt), "rag": StackedRegionAdjacencyGraph(self.pathToSeg)}

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
        uGt = nodeGt[uvIds[:, 0]]
        vGt = nodeGt[uvIds[:, 1]]
        edgeGt = (uGt != vGt).astype('uint8')
        assert (np.unique(edgeGt) == np.array([0, 1])).all(), str(np.unique(edgeGt))
        assert edgeGt.shape[0] == uvIds.shape[0]
        self.output().write(edgeGt)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        def_str = 'modified' if PipelineParameter().defectPipeline else 'standard'
        save_path = os.path.join(PipelineParameter().cache, "EdgeGroundtruth_%s_%s.h5" % (segFile, def_str))
        return HDF5DataTarget(save_path)


class LearnClassifierFromGt(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToGt  = luigi.ListParameter()

    def requires(self):
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        n_inputs = len(self.pathsToSeg)

        if n_inputs == 1:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from single input")
            feature_tasks = get_local_features()
            return_tasks = {
                "gt": EdgeGroundtruth(self.pathsToSeg[0], self.pathsToGt[0]),
                "features": feature_tasks,
                "rag": StackedRegionAdjacencyGraph(self.pathsToSeg[0])
            }
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from %i inputs" % n_inputs)
            feature_tasks = get_local_features_for_multiinp()
            assert len(feature_tasks) == n_inputs
            return_tasks = {
                "gt": [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in xrange(n_inputs)],
                "features": feature_tasks,
                "rag": [StackedRegionAdjacencyGraph(segp) for segp in self.pathsToSeg]
            }
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
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from single input for xy and z edges separately."
            )
            self._learn_classifier_from_single_input_xy(gt, feature_tasks, transitionEdge)
            self._learn_classifier_from_single_input_z(gt, feature_tasks, transitionEdge, skipTransition)

        else:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for all edges.")
            features = np.concatenate(
                [feat.read([0, 0], feat.shape(key), key)
                 for key in ('features_xy', 'features_z') for feat in feature_tasks],
                axis=1
            )

            if PipelineParameter().defectPipeline:
                gt = gt[:skipTransition]

            assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
            classifier = RandomForest(
                features,
                gt,
                n_trees=PipelineParameter().nTrees,
                max_depth=PipelineParameter().maxDepth,
                n_threads=PipelineParameter().nThreads
            )
            classifier.write(str(self.output().path), 'rf_joined')

        if PipelineParameter().defectPipeline:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from single input for skip edges (defects)."
            )
            self._learn_classifier_from_single_input_defects(gt, feature_tasks, skipTransition)

    def _learn_classifier_from_single_input_xy(self, gt, feature_tasks, transitionEdge):
        gt = gt[:transitionEdge]
        features = np.concatenate(
            [feat_task.read([0, 0], feat_task.shape('features_xy'), 'features_xy') for feat_task in feature_tasks],
            axis=1
        )
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = RandomForest(
            features,
            gt,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_xy')

    # if we learn with defects, we only consider the z edges that are not skip edges here
    def _learn_classifier_from_single_input_z(self, gt, feature_tasks, transitionEdge, skipTransition):

        gt = gt[transitionEdge:skipTransition] if PipelineParameter().defectPipeline else gt[transitionEdge:]
        features = np.concatenate(
            [feat_task.read([0, 0], feat_task.shape('features_z'), 'features_z') for feat_task in feature_tasks],
            axis=1
        )
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = RandomForest(
            features,
            gt,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_z')

    # if we learn with defects, we only consider the skip edges here
    def _learn_classifier_from_single_input_defects(self, gt, feature_tasks, skipTransition):
        assert PipelineParameter().defectPipeline
        gt = gt[skipTransition:]
        features = np.concatenate(
            [feat_task.read([0, 0], feat_task.shape('features_skip'), 'features_skip') for feat_task in feature_tasks],
            axis=1
        )
        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = RandomForest(
            features,
            gt,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_skip')

    def _learn_classifier_from_multiple_inputs(self, rag, gt, feature_tasks):

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classifier for multiple inputs for xy and z edges separately."
            )
            self._learn_classifier_from_multiple_inputs_xy(rag, gt, feature_tasks)
            self._learn_classifier_from_multiple_inputs_z(rag, gt, feature_tasks)

        else:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classifier for multiple inputs for all edges jointly."
            )
            self._learn_classifier_from_multiple_inputs_all(rag, gt, feature_tasks)

        if PipelineParameter().defectPipeline:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from multiple inputs for skip edges (defects)."
            )
            self._learn_classifier_from_multiple_inputs_defects(rag, gt, feature_tasks)

    def _learn_classifier_from_multiple_inputs_xy(self, rag, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for xy called.")
        features = []
        gts = []

        for i in xrange(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i].read()
            rag_i = rag[i]

            transitionEdge = rag_i.readKey('totalNumberOfInSliceEdges')
            gt_i = gt_i[:transitionEdge]
            features_i = np.concatenate(
                [feat_task.read([0, 0], feat_task.shape('features_xy'), 'features_xy') for feat_task in feat_tasks_i],
                axis=1
            )
            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(
            features,
            gts,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_xy')

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
                [feat_task.read([0, 0], feat_task.shape('features_z'), 'features_z') for feat_task in feat_tasks_i],
                axis=1
            )

            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(
            features,
            gts,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_z')

    def _learn_classifier_from_multiple_inputs_all(self, rag, gt, feature_tasks):
        features = []
        gts = []
        for i in xrange(len(gt)):

            rag_i = rag[i]
            if PipelineParameter().defectPipeline:
                if self.input()["modified_adjacency"][i].read("has_defects"):
                    skipTransition = rag_i.readKey('numberOfEdges') - \
                        self.input()["modified_adjacency"][i].read("delete_edges").shape[0]
                else:
                    skipTransition = rag_i.readKey('numberOfEdges')
            else:
                skipTransition = rag_i.readKey('numberOfEdges')

            features.append(np.concatenate(
                [feat.read([0, 0], feat.shape(key), key)
                 for key in ('features_xy', 'features_z') for feat in feature_tasks[i]],
                axis=1)
            )
            gts.append(gt[i].read()[:skipTransition])

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(
            features,
            gts,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_joined')

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
            skipTransition = rag_i.readKey('numberOfEdges') - mod_i.read("delete_edges").shape[0]

            gt_i = gt_i[skipTransition:]
            features_i = np.concatenate(
                [feat_task.read([0, 0], feat_task.shape('features_skip'), 'features_skip')
                 for feat_task in feat_tasks_i],
                axis=1
            )

            features.append(features_i)
            gts.append(gt_i)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(
            features,
            gts,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_skip')

    def output(self):
        ninp_str = "SingleInput" if (len(self.pathsToSeg) == 1) else "MultipleInput"
        save_path = os.path.join(PipelineParameter().cache, "LearnClassifierFromGt_%s" % ninp_str)
        return FolderTarget(save_path) if use_sklearn else HDF5DataTarget(save_path)
