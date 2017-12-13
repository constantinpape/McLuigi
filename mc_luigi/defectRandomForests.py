from . learningTasks import RandomForest

import os
import luigi
import numpy as np
import logging

# import the proper nifty version
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi.graph.rag as nrag

from .dataTasks import StackedRegionAdjacencyGraph, InputData
from .tools import config_logger, run_decorator
from .featureTasks import RegionNodeFeatures
from .customTargets import HDF5DataTarget, FolderTarget
from .pipelineParameter import PipelineParameter

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


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
