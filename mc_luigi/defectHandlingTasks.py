from __future__ import print_function, division

# Multicut Pipeline implemented with luigi
# Taksks for defect and handling
import luigi

from .customTargets import HDF5DataTarget
from .dataTasks import ExternalSegmentation, StackedRegionAdjacencyGraph
from .defectDetectionTasks import DefectSliceDetection
from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator, get_unique_rows

import logging

import os
import numpy as np

from concurrent import futures

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


# map defect patches to overlapping superpixels
class DefectsToNodes(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'seg': ExternalSegmentation(self.pathToSeg),
                'defects': DefectSliceDetection(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        seg = inp['seg']
        seg.open()
        defects = inp['defects']
        defects.open()

        assert seg.shape() == defects.shape()

        ny = seg.shape()[1]
        nx = seg.shape()[2]

        def defects_to_nodes_z(z):
            slice_begin = [z, 0, 0]
            slice_end   = [z + 1, ny, nx]
            defect_mask = defects.read(slice_begin, slice_end)
            if 1 in defect_mask:
                print(z)
                seg_z = seg.read(slice_begin, slice_end)
                where_defect = defect_mask == 1
                defect_nodes_slice = np.unique(seg_z[where_defect])
                return list(defect_nodes_slice), len(defect_nodes_slice) * [z]
            else:
                return [], []

        # non-parallel for debugging
        # defect_nodes = []
        # nodes_z = []
        # for z in range(seg.shape[0]):
        #    res = defects_to_nodes_z(z)
        #    defect_nodes.extend(res[0])
        #    nodes_z.extend(res[1])

        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = [executor.submit(defects_to_nodes_z, z) for z in range(seg.shape()[0])]
            defect_nodes = []
            nodes_z      = []
            for fut in tasks:
                nodes, zz = fut.result()
                if nodes:
                    defect_nodes.extend(nodes)
                    nodes_z.extend(zz)

        assert len(defect_nodes) == len(nodes_z)

        workflow_logger.info("DefectsToNodes: found %i defected nodes" % (len(defect_nodes)))

        self.output().write(np.array(defect_nodes, dtype='uint32'), 'defect_nodes')
        self.output().write(np.array(nodes_z, dtype='uint32'), 'nodes_z')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(
            PipelineParameter().cache,
            "DefectsToNodes_%s_nBins%i_binThreshold%i.h5" % (
                segFile,
                PipelineParameter().nBinsSliceStatistics,
                PipelineParameter().binThreshold
            )
        )
        return HDF5DataTarget(save_path)


# modify the rag to isolate defected superpixels
# introduce skip edges over the defected superpixels in z
class ModifiedAdjacency(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'rag': StackedRegionAdjacencyGraph(self.pathToSeg),
                'seg': ExternalSegmentation(self.pathToSeg),
                'defect_nodes': DefectsToNodes(self.pathToSeg)}

    # compute and save the skip edges as tuples (n1,n2) as well
    # as the resulting modified adjacency (as plain graph)
    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()
        defect_nodes = inp['defect_nodes'].read('defect_nodes')
        nodes_z = inp['defect_nodes'].read('nodes_z')
        seg = inp['seg']
        seg.open()

        # make sure that z is monotonically increasing (not strictly!)
        assert np.all(np.diff(nodes_z.astype(int)) >= 0), "Defected slice index is not increasing monotonically!"

        # loop over the defect nodes and find the skip edges,
        # compute the modify adjacency

        delete_edges = []  # the z-edges between defected and non-defected nodes that are deleted from the graph
        # the xy-edges between defected and non-defected nodes, that will be set to maximally repulsive weights
        ignore_edges = []

        skip_edges   = []  # the skip edges that run over the defects in z
        skip_ranges  = []  # z-distance of the skip edges
        skip_starts  = []  # starting slices of the skip edges

        # get the original rag adjacency
        uv_ids = rag.uvIds()
        n_nodes = uv_ids.max() + 1

        defect_slices = np.unique(nodes_z)

        defect_node_dict = {int(z): list(defect_nodes[nodes_z == z].astype(int)) for z in defect_slices}

        # FIXME TODO can't do this here once we have individual defect patches
        consecutive_defect_slices = np.split(defect_slices, np.where(np.diff(defect_slices) != 1)[0] + 1)
        has_lower_defect_list = []
        for consec in consecutive_defect_slices:
            if len(consec) > 1:
                has_lower_defect_list.extend(consec[1:])
        print("Slices with lower defect slice:", has_lower_defect_list)

        def get_skip_edges_from_nifty(z, i):
            has_lower_defect = z in has_lower_defect_list
            delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = nrag.getSkipEdgesForSlice(
                rag,
                int(z),
                defect_node_dict,
                has_lower_defect
            )
            print('Finished processing slice', z, ':', i, '/', len(defect_slices))
            return z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z

        # non-parallel for debugging
        # for i, z in enumerate(defect_slices):
        #    z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = get_skip_edges_from_nifty(z,i)

        #    delete_edges.extend(delete_edges_z)
        #    ignore_edges.extend(ignore_edges_z)

        #    assert len(skip_edges_z) == len(skip_ranges_z)
        #    skip_edges.extend(skip_edges_z)
        #    skip_ranges.extend(skip_ranges_z)
        #    skip_starts.extend(len(skip_edges_z) * [z-1])

        # parallel
        with futures.ThreadPoolExecutor(max_workers=PipelineParameter().nThreads) as executor:
            tasks = [executor.submit(get_skip_edges_from_nifty, z, i) for i, z in enumerate(defect_slices)]
            for task in tasks:

                z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = task.result()

                delete_edges.extend(delete_edges_z)
                ignore_edges.extend(ignore_edges_z)

                assert len(skip_edges_z) == len(skip_ranges_z)
                skip_edges.extend(skip_edges_z)
                skip_ranges.extend(skip_ranges_z)
                skip_starts.extend(len(skip_edges_z) * [z - 1])

        delete_edges = np.unique(delete_edges).astype(np.uint32)
        uv_ids = np.delete(uv_ids, delete_edges, axis=0)
        workflow_logger.info("ModifiedAdjacency: deleted %i z-edges due to defects" % len(delete_edges))

        skip_edges = np.array(skip_edges, dtype="uint32")
        skips_before_unique = skip_edges.shape[0]

        # TODO need to take care of corner cases when we have delete edges but no skip edges...
        if skip_edges.size:
            skip_edges, idx = get_unique_rows(skip_edges, return_index=True)
            workflow_logger.info(
                "ModifiedAdjacency: Removed %i duplicate skip edges" % (skips_before_unique - skip_edges.shape[0])
            )

            skip_ranges = np.array(skip_ranges, dtype="uint32")
            skip_ranges = skip_ranges[idx]
            skip_starts = np.array(skip_starts, dtype="uint32")
            skip_starts = skip_starts[idx]
            assert skip_edges.shape[0] == skip_ranges.shape[0]
            assert skip_starts.shape[0] == skip_ranges.shape[0]
            workflow_logger.info("ModifiedAdjacency: introduced %i skip edges due to defects" % len(skip_edges))

            # reorder the skip edges s.t. skip_starts are monotonically increasing
            sort_indices = np.argsort(skip_starts)
            skip_edges = skip_edges[sort_indices]
            skip_ranges = skip_ranges[sort_indices]
            skip_starts = skip_starts[sort_indices]
            # make sure that z is monotonically increasing (not strictly!)
            assert np.all(np.diff(skip_starts.astype(int)) >= 0), "Start index of skip edges must increase."

            ignore_edges = np.unique(ignore_edges).astype(np.uint32)
            workflow_logger.info("ModifiedAdjacency: found %i ignore edges due to defects" % len(ignore_edges))

            # create the modified adjacency
            modified_adjacency = nifty.graph.UndirectedGraph(int(n_nodes))
            # insert original remaining edges
            modified_adjacency.insertEdges(uv_ids)
            # insert skip edges
            if skip_edges.size:
                modified_adjacency.insertEdges(skip_edges)
            n_edges_modified = modified_adjacency.numberOfEdges
            assert n_edges_modified == rag.numberOfEdges - delete_edges.shape[0] + skip_edges.shape[0], "%i, %i" % (
                n_edges_modified,
                rag.numberOfEdges - delete_edges.shape[0] + skip_edges.shape[0]
            )
            workflow_logger.info(
                "ModifiedAdjacency: Total number of edges in modified adjacency: %i" % n_edges_modified
            )
            modified_adjacency = modified_adjacency.serialize()
            out = self.output()
            out.write(True, "has_defects")
            out.write(modified_adjacency, "modified_adjacency")
            out.write(skip_edges, "skip_edges")
            out.write(skip_starts, "skip_starts")
            out.write(skip_ranges, "skip_ranges")
            out.write(delete_edges, "delete_edges")
            out.write(ignore_edges, "ignore_edges")
            out.write(n_edges_modified, "n_edges_modified")

        else:
            workflow_logger.info("ModifiedAdjacency: No defects, writing dummy data")
            self.output().write(False, "has_defects")

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "ModifiedAdjacency_%s.h5" % segFile)
        return HDF5DataTarget(save_path)


class SkipEdgeLengths(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'rag': StackedRegionAdjacencyGraph(self.pathToSeg),
                'modified_adjacency': ModifiedAdjacency(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()
        mod_adjacency = inp['modified_adjacency']

        if not mod_adjacency.read('has_defects'):
            self.output().write(np.empty(0), 'data')
            return

        skip_edges = mod_adjacency.read('skip_edges')
        skip_ranges = mod_adjacency.read('skip_ranges')
        skip_starts = mod_adjacency.read('skip_starts')

        skip_lens = nrag.getSkipEdgeLengths(
            rag,
            # skip_edges need to be passed as a list of pairs!
            [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges],
            list(skip_ranges),
            list(skip_starts),
            PipelineParameter().nThreads
        )
        skip_lens = np.array(skip_lens, dtype='uint32')
        assert skip_lens.shape[0] == skip_edges.shape[0]
        workflow_logger.info(
            "SkipEdgeLengths: computed lens in range %i to %i" % (skip_lens.min(), skip_lens.max())
        )

        self.output().write(skip_lens, 'data')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "SkipEdgeLengths_%s.h5" % segFile)
        return HDF5DataTarget(save_path)
