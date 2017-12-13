from __future__ import print_function, division

# Multicut Pipeline implemented with luigi
# Tasks for learning and predicting random forests

import luigi

from .taskSelection import get_local_features, get_local_features_for_multiinp
from .customTargets import HDF5DataTarget, VolumeTarget, FolderTarget
from .dataTasks import DenseGroundtruth, StackedRegionAdjacencyGraph, InputData
from .defectHandlingTasks import ModifiedAdjacency
from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator
from .featureTasks import RegionNodeFeatures

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
    import pickle
    workflow_logger.debug("Using sklearn random forest")
except ImportError:
    RFType = vigra.learning.RandomForest3
    use_sklearn = False
    workflow_logger.debug("Using vigra random forest 3")


# wrapper for sklearn / random forest
class RandomForest(object):

    def __init__(self,
                 train_data,
                 train_labels,
                 n_trees,
                 n_threads,
                 max_depth=None):
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
            assert os.path.exists(save_path), save_path
            with open(save_path, 'rb') as f:
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
            with open(save_path, 'wb') as f:
                pickle.dump(self.rf, f)
        else:
            save_path = file_path + ".h5"
            self.rf.writeHDF5(file_path, key)

    # TODO for vigra
    @property
    def n_features(self):
        return self.rf.n_features_


class EdgeProbabilities(luigi.Task):

    pathToSeg        = luigi.Parameter()
    pathToClassifier = luigi.Parameter()

    def requires(self):
        return_tasks = {"features": get_local_features(),
                        "rag": StackedRegionAdjacencyGraph(self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            return_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return return_tasks

    @run_decorator
    def run(self):
        assert os.path.exists(self.pathToClassifier), self.pathToClassifier
        inp = self.input()
        feature_tasks = inp["features"]

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

        out = self.output()
        # 262144 = chunk size (64**3)
        out.open('data', shape=(n_edges,), chunks=(min(262144, n_edges),), dtype='float32')
        self._predict(feature_tasks, out)
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

        # find how many features we have for the different feature types
        n_feats_xy = 0
        n_feats_z  = 0
        n_feats_skip = 0
        for feat in feature_tasks:

            xy_key = 'features_xy'
            if xy_key in feat:
                feat.open(xy_key)
                n_feats_xy += feat.shape(xy_key)[1]

            z_key = 'features_z'
            if z_key in feat:
                feat.open(z_key)
                n_feats_z += feat.shape(z_key)[1]

            if has_defects:
                skip_key = 'features_skip'
                if skip_key in feat:
                    feat.open(skip_key)
                    n_feats_skip += feat.shape(skip_key)[1]

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
                self._predict_out_of_core(classifier,
                                          feature_tasks,
                                          out,
                                          feat_type,
                                          n_edges_type,
                                          n_feats_type,
                                          start_type)
            else:
                workflow_logger.info(
                    "EdgeProbabilities: predicting %s edges with seperate claissifier in core." % feat_type
                    if PipelineParameter().separateEdgeClassification else
                    "EdgeProbabilities: predicting %s edges with joint classifier in core" % feat_type
                )
                self._predict_in_core(classifier,
                                      feature_tasks,
                                      out,
                                      feat_type,
                                      n_edges_type,
                                      n_feats_type,
                                      start_type)

    # In core prediction for edge type
    def _predict_in_core(self,
                         classifier,
                         feature_tasks,
                         out,
                         feat_type,
                         n_edges,
                         n_feats,
                         start):

        # read all features of this type in one chunk
        offset = 0
        features = np.zeros((n_edges, n_feats), dtype='float32')

        for ii, feat in enumerate(feature_tasks):
            if feat_type in feat:
                this_feats = feat.read((0, 0), feat.shape(feat_type), feat_type)
                features[:, offset:offset + this_feats.shape[1]] = this_feats
                offset += this_feats.shape[1]

        assert features.shape[1] == classifier.n_features, \
            "Number of input and rf features do not match for %s: %i, %i" % (
                feat_type, features.shape[1], classifier.n_features)
        probs = classifier.predict_probabilities(features, PipelineParameter().nThreads)[:, 1]
        out.write([start], probs)

    # Out of core prediction for edge type
    def _predict_out_of_core(self,
                             classifier,
                             feature_tasks,
                             out,
                             feat_type,
                             n_edges,
                             n_feats,
                             start):
        n_sub_feats = PipelineParameter().nFeatureChunks
        assert n_sub_feats > 1, str(n_sub_feats)

        def predict_subfeats(sub_feat_id):
            print(sub_feat_id, '/', n_sub_feats)
            start_index = int(float(sub_feat_id) / n_sub_feats * n_edges)
            end_index   = int(float(sub_feat_id + 1) / n_sub_feats * n_edges)
            if sub_feat_id == n_sub_feats:
                end_index = n_edges

            sub_feats = []
            for feat in feature_tasks:
                if feat_type in feat:
                    sub_feats.append(feat.read((start_index, 0),
                                               (end_index, feat.shape(feat_type)[1]),
                                               feat_type))
            sub_feats = np.concatenate(sub_feats, axis=1)

            read_start = start_index + start

            probs = classifier.predict_probabilities(sub_feats, 1)[:, 1]
            out.write([read_start], probs)
            return True

        n_workers = PipelineParameter().nThreads
        # n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = [executor.submit(predict_subfeats, sub_feat_id) for sub_feat_id in range(n_sub_feats)]
            [t.result() for t in tasks]

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "EdgeProbabilities%s_%s.h5" % (
                "Separate" if PipelineParameter().separateEdgeClassification else "Joint",
                "modified" if PipelineParameter().defectPipeline else "standard"
            )
        )
        save_path += VolumeTarget.file_ending()
        return VolumeTarget(save_path)


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

        # check for defects and load the correspomding uv-ids
        has_defects = False
        if PipelineParameter().defectPipeline:
            if inp["modified_adjacency"].read("has_defects"):
                has_defects = True

        if has_defects:
            mod_adjacency = nifty.graph.UndirectedGraph()
            mod_adjacency.deserialize(inp["modified_adjacency"].read("modified_adjacency"))
            uv_ids = mod_adjacency.uvIds()
        else:
            uv_ids = rag.uvIds()

        out = self.output()
        u_gt, v_gt = self._compute_edge_gt(gt, rag, uv_ids, has_defects, inp, out)

        # check if we have an ignore label in the groundtruth and mask the labels accordingly
        if PipelineParameter().ignoreLabel != -1:
            self._compute_label_masks(u_gt, v_gt, rag, has_defects, inp, out)

        gt.close()

    def _compute_edge_gt(self, gt, rag, uv_ids, has_defects, inp, out):

        node_gt = nrag.gridRagAccumulateLabels(rag, gt.get())
        u_gt = node_gt[uv_ids[:, 0]]
        v_gt = node_gt[uv_ids[:, 1]]
        edge_gt = (u_gt != v_gt).astype('uint8')

        assert (np.unique(edge_gt) == np.array([0, 1])).all(), str(np.unique(edge_gt))
        assert edge_gt.shape[0] == uv_ids.shape[0]

        # write the edge gts for all the different edge types
        edge_transition = rag.totalNumberOfInSliceEdges

        out.write(edge_gt, 'edge_gt')
        out.write(edge_gt[:edge_transition], 'edge_gt_xy')
        if has_defects:
            mod_adjacency = inp["modified_adjacency"]
            skip_transition = rag.numberOfEdges - mod_adjacency.read("delete_edges").shape[0]

            out.write(edge_gt[edge_transition:skip_transition], 'edge_gt_z')
            out.write(edge_gt[skip_transition:], 'edge_gt_skip')
        else:
            out.write(edge_gt[edge_transition:], 'edge_gt_z')

        return u_gt, v_gt

    def _compute_label_masks(self, u_gt, v_gt, rag, has_defects, inp, out):

        ignore_label = PipelineParameter().ignoreLabel
        assert ignore_label != -1
        label_mask = np.logical_not(
            np.logical_or((u_gt == ignore_label), (v_gt == ignore_label))
        )

        # write the label masks for all edge types
        edge_transition = rag.totalNumberOfInSliceEdges
        label_mask_xy = label_mask[:edge_transition]
        out.write(label_mask_xy, 'label_mask_xy')

        if has_defects:

            mod_adjacency = inp["modified_adjacency"]
            skip_transition = rag.numberOfEdges - mod_adjacency.read("delete_edges").shape[0]

            label_mask_z = label_mask[edge_transition:skip_transition]
            out.write(label_mask_z, 'label_mask_z')

            label_mask_skip = label_mask[skip_transition:]
            out.write(label_mask_skip, 'label_mask_skip')

            out.write(label_mask[:skip_transition], 'label_mask')

        else:
            label_mask_z = label_mask[edge_transition:]
            out.write(label_mask_z, 'label_mask_z')
            out.write(label_mask, 'label_mask')

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
            return_tasks = {"gt": EdgeGroundtruth(self.pathsToSeg[0], self.pathsToGt[0]),
                            "features": feature_tasks}
        else:
            workflow_logger.info("LearnClassifierFromGt: learning classifier from %i inputs" % n_inputs)
            feature_tasks = get_local_features_for_multiinp()
            assert len(feature_tasks) == n_inputs
            return_tasks = {"gt": [EdgeGroundtruth(self.pathsToSeg[i], self.pathsToGt[i]) for i in range(n_inputs)],
                            "features": feature_tasks}
        if PipelineParameter().defectPipeline:
            return_tasks['modified_adjacency'] = [ModifiedAdjacency(self.pathsToSeg[i]) for i in range(n_inputs)]
        return return_tasks

    @run_decorator
    def run(self):
        inp = self.input()
        gt = inp["gt"]
        feature_tasks = inp["features"]

        # open all feature tasks
        for feat in feature_tasks:
            [feat.open(key) for key in feat.keys_on_filesystem()]

        n_inputs = len(self.pathsToSeg)

        if n_inputs > 1:
            n_inputs = len(gt)
            assert n_inputs == len(feature_tasks)

            workflow_logger.info("LearnClassifierFromGt: call learning classifier for %i inputs" % n_inputs)
            self._learn_classifier_from_multiple_inputs(gt, feature_tasks)

        else:
            workflow_logger.info("LearnClassifierFromGt: call learning classifier for single inputs")
            self._learn_classifier_from_single_input(gt, feature_tasks)

    def _learn_classifier_from_single_input(self, gt, feature_tasks):
        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from single input for xy and z edges separately."
            )
            self._learn_classifier_from_single_input_xy(gt, feature_tasks)
            self._learn_classifier_from_single_input_z(gt, feature_tasks)

        else:
            workflow_logger.info("LearnClassifierFromGt: learning classfier from single input for all edges.")
            features = []
            for feat in feature_tasks:
                features.append(np.concatenate([feat.read(start=(0, 0), stop=feat.shape(key=key), key=key)
                                                for key in ('features_xy', 'features_z')],
                                               axis=0))
            features = np.concatenate(features, axis=1)

            edge_gt = gt.read('edge_gt')

            # if we have an ignore mask, mask the features and labels for which we don't have a label
            if PipelineParameter().ignoreLabel != -1:
                mask = gt.read('label_mask')
                features = features[mask]
                edge_gt = edge_gt[mask]

            assert features.shape[0] == edge_gt.shape[0], str(features.shape[0]) + " , " + str(edge_gt.shape[0])
            classifier = RandomForest(features,
                                      edge_gt,
                                      n_trees=PipelineParameter().nTrees,
                                      max_depth=PipelineParameter().maxDepth,
                                      n_threads=PipelineParameter().nThreads)
            classifier.write(str(self.output().path), 'rf_joined')

        if PipelineParameter().defectPipeline:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from single input for skip edges (defects)."
            )
            self._learn_classifier_from_single_input_defects(gt, feature_tasks)

    def _learn_classifier_from_single_input_xy(self, gt, feature_tasks):
        edge_gt = gt.read('edge_gt_xy')

        features = np.concatenate([feat_task.read((0, 0), feat_task.shape(key='features_xy'), key='features_xy')
                                   for feat_task in feature_tasks if 'features_xy' in feat_task.keys()],
                                  axis=1)

        if PipelineParameter().ignoreLabel != -1:
            mask = gt.read('label_mask_xy')
            features = features[mask]
            edge_gt = edge_gt[mask]

        assert features.shape[0] == edge_gt.shape[0], str(features.shape[0]) + " , " + str(edge_gt.shape[0])
        classifier = RandomForest(features,
                                  edge_gt,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_xy')

    def _learn_classifier_from_single_input_z(self, gt, feature_tasks):
        edge_gt = gt.read('edge_gt_z')

        features = np.concatenate([feat_task.read((0, 0), feat_task.shape(key='features_z'), key='features_z')
                                   for feat_task in feature_tasks if 'features_z' in feat_task.keys()],
                                  axis=1)

        if PipelineParameter().ignoreLabel != -1:
            mask = gt.read('label_mask_z')
            features = features[mask]
            edge_gt = edge_gt[mask]

        assert features.shape[0] == edge_gt.shape[0], "%i , %i" % (features.shape[0], edge_gt.shape[0])
        classifier = RandomForest(features,
                                  edge_gt,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_z')

    # FIXME defects currently not supported
    def _learn_classifier_from_single_input_defects(self, gt, feature_tasks):
        assert PipelineParameter().defectPipeline
        edge_gt = gt.read('edge_gt_skip')

        features = []
        for feat_task in feature_tasks:
            feat_path = os.path.join(feat_task.path, 'features_skip.h5')
            if os.path.exists(feat_path):
                with h5py.File(feat_path) as f:
                    features.append(f['data'][:])
        features = np.concatenate(features, axis=1)

        if PipelineParameter().ignoreLabel != -1:
            mask = gt.read('label_mask_skip')
            features = features[mask]
            edge_gt = edge_gt[mask]

        assert features.shape[0] == gt.shape[0], str(features.shape[0]) + " , " + str(gt.shape[0])
        classifier = RandomForest(
            features,
            edge_gt,
            n_trees=PipelineParameter().nTrees,
            max_depth=PipelineParameter().maxDepth,
            n_threads=PipelineParameter().nThreads
        )
        classifier.write(str(self.output().path), 'rf_features_skip')

    def _learn_classifier_from_multiple_inputs(self, gt, feature_tasks):

        if PipelineParameter().separateEdgeClassification:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classifier for multiple inputs for xy and z edges separately."
            )
            self._learn_classifier_from_multiple_inputs_xy(gt, feature_tasks)
            self._learn_classifier_from_multiple_inputs_z(gt, feature_tasks)

        else:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classifier for multiple inputs for all edges jointly."
            )
            self._learn_classifier_from_multiple_inputs_all(gt, feature_tasks)

        if PipelineParameter().defectPipeline:
            workflow_logger.info(
                "LearnClassifierFromGt: learning classfier from multiple inputs for skip edges (defects)."
            )
            self._learn_classifier_from_multiple_inputs_defects(gt, feature_tasks)

    def _learn_classifier_from_multiple_inputs_xy(self, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for xy called.")
        features = []
        gts = []

        for i in range(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i]

            features_i = np.concatenate([feat_task.read((0, 0), feat_task.shape(key='features_xy'), key='features_xy')
                                         for feat_task in feat_tasks_i if 'features_xy' in feat_task_i.keys()],
                                        axis=1)
            edge_gt = gt_i.read('edge_gt_xy')

            if PipelineParameter().ignoreLabel != -1:
                mask = gt_i.read('label_mask_xy')
                features_i = features_i[mask]
                edge_gt = edge_gt[mask]

            features.append(features_i)
            gts.append(edge_gt)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(features, gts,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_xy')

    def _learn_classifier_from_multiple_inputs_z(self, gt, feature_tasks):

        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for z called.")
        features = []
        gts = []

        for i in range(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i]
            features_i = np.concatenate([feat_task.read((0, 0), feat_task.shape(key='features_z'), key='features_z')
                                         for feat_task in feat_tasks_i if 'features_z' in feat_task_i.keys()],
                                        axis=1)
            edge_gt = gt_i.read('edge_gt_z')

            if PipelineParameter().ignoreLabel != -1:
                mask = gt_i.read('label_mask_z')
                assert len(mask) == len(features_i), "%s, %s" % (str(mask.shape), str(features_i.shape))
                assert len(mask) == len(edge_gt), "%s, %s" % (str(mask.shape), str(edge_gt.shape))
                features_i = features_i[mask]
                edge_gt = edge_gt[mask]

            features.append(features_i)
            gts.append(edge_gt)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(features, gts,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_z')

    def _learn_classifier_from_multiple_inputs_all(self, gt, feature_tasks):

        features = []
        gts = []

        for i in range(len(gt)):

            gt_i = gt[i]
            edge_gt = gt_i.read('edge_gt')

            features_i = []
            for feat_task in feature_tasks[i]:
                features_i.append(np.concatenate([feat_task.read((0, 0), feat_task.shape(key=key), key=key)
                                                  for key in ('features_xy', 'features_z')],
                                                axis=1))
            features_i = np.concatenate(features_i, axis=1)

            if PipelineParameter().ignoreLabel != -1:
                mask = gt_i.read('label_mask')
                features_i = features_i[mask]
                edge_gt = edge_gt[mask]

            features.append(features_i)
            gts.append(edge_gt)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(features, gts,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_joined')

    # FIXME defects currently not supported
    def _learn_classifier_from_multiple_inputs_defects(self, gt, feature_tasks):
        workflow_logger.info("LearnClassifierFromGt: learning classifier for multiple inputs for defects called.")
        assert PipelineParameter().defectPipeline
        features = []
        gts = []

        for i in range(len(gt)):
            feat_tasks_i = feature_tasks[i]
            gt_i = gt[i]

            # if we learn with defects, we only keep the z edges that are not skip edges
            mod_i = self.input()["modified_adjacency"][i]
            if not mod_i.read("has_defects"):
                continue

            edge_gt = gt_i.read('edge_gt_skip')
            features_i = []
            for feat_task in feat_tasks_i:
                feat_path = os.path.join(feat_task.path, 'features_skip.h5')
                if os.path.exists(feat_path):
                    with h5py.File(feat_path) as f:
                        features_i.append(f['data'][:])
            features_i = np.concatenate(features_i, axis=1)

            if PipelineParameter().ignoreLabel != -1:
                mask = gt_i.read('label_mask_skip')
                features_i = features_i[mask]
                edge_gt = edge_gt[mask]

            features.append(features_i)
            gts.append(edge_gt)

        features = np.concatenate(features, axis=0)
        gts      = np.concatenate(gts)
        assert features.shape[0] == gts.shape[0], str(features.shape[0]) + " , " + str(gts.shape[0])
        classifier = RandomForest(features, gts,
                                  n_trees=PipelineParameter().nTrees,
                                  max_depth=PipelineParameter().maxDepth,
                                  n_threads=PipelineParameter().nThreads)
        classifier.write(str(self.output().path), 'rf_features_skip')

    def output(self):
        ninp_str = "SingleInput" if (len(self.pathsToSeg) == 1) else "MultipleInput"
        save_path = os.path.join(PipelineParameter().cache, "LearnClassifierFromGt_%s" % ninp_str)
        return FolderTarget(save_path) if use_sklearn else HDF5DataTarget(save_path)


class DefectNodeGroundtruth(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToDefectGt = luigi.Parameter()

    def requires(self):
        return{
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "defect_gt": InputData(self.pathToDefectGt, dtype='uint8')
        }

    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()
        defect_gt = inp['defect_gt']
        defect_gt.open()

        node_labels = nrag.gridRagAccumulateLabels(rag, defect_gt.get())
        assert (np.unique(node_labels) == np.array([0, 1])).all(), str(np.unique(node_labels))
        self.output().write(node_labels)

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        save_path = "DefectNodeGroundtruth_%s.h5" % seg_file
        return HDF5DataTarget(save_path)


class LearnDefectRandomForest(luigi.Task):

    pathsToSeg = luigi.ListParameter()
    pathsToDefectGt = luigi.ListParameter()

    def requires(self):
        assert len(self.pathsToSeg) == len(self.pathsToGt)
        n_inputs = len(self.pathsToSeg)
        inputs = PipelineParameter().inputs

        if n_inputs == 1:
            raw_path = inputs['data'][0]
            return {
                'gt': DefectNodeGroundtruth(self.pathsToSeg[0], self.pathsToDefectGt[0]),
                'feats': RegionNodeFeatures(self.pathsToSeg[0], raw_path)
            }
        else:
            inp_paths = inputs['data']
            assert n_inputs % inp_paths == 0
            inp_per_seg = len(inp_paths) // n_inputs
            return {
                'gt': [DefectNodeGroundtruth(self.pathsToSeg[i], self.pathsToDefectGt[i]) for i in range(n_inputs)],
                'feats': [RegionNodeFeatures(self.pathToSeg[i], inp_paths[inp_per_seg * i]) for i in range(n_inputs)]
            }

    @run_decorator
    def run(self):
        if(self.pathsToSeg) > 1:
            self._learn_defect_rf_multi_input()
        else:
            self._learn_defect_rf_single_input()

    def _learn_defect_rf_multi_input(self):
        inp = self.input()
        gts = inp['gt']
        feats = inp['feats']
        assert len(gts) == len(feats)
        features = []
        labels = []
        for i, gt in enumerate(gts):
            this_gt = gt.read()
            this_feats = feats[i].read([0, 0], feats[i].shape)
            assert len(this_gt) == len(this_feats), "%i, %i" % (len(this_gt), len(this_feats))
            features.append(this_feats)
            labels.append(this_gt)
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        rf = RandomForest(
            features, labels,
            n_trees=PipelineParameter().nTrees,
            n_threads=PipelineParameter().nThreads
        )
        rf.write(str(self.output().path), 'rf')

    def _learn_defect_rf_single_input(self):
        inp = self.input()
        gt = inp['gt'].read()
        feats = inp['feats']
        feats = feats.readSubarray([0, 0], feats.shape)
        assert len(gt) == len(feats), "%i, %i" % (len(gt), len(feats))
        rf = RandomForest(
            feats, gt,
            n_trees=PipelineParameter().nTrees,
            n_threads=PipelineParameter().nThreads
        )
        rf.write(str(self.output().path), 'rf')

    def output(self):
        save_path = 'LearnDefectRandomForest_%s' % (
            'multi_input' if len(self.pathsToSeg) > 1 else 'single_input',
        )
        return FolderTarget(save_path)
