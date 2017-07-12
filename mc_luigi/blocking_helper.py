import luigi
from tools import run_decorator, config_logger, cartesian
from customTargets import HDF5DataTarget
from dataTasks import ExternalSegmentation
from pipelineParameter import PipelineParameter
from defectDetectionTasks import DefectSliceDetection
from tools import find_matching_row_indices

import numpy as np
import os
import vigra
import logging

# import the proper nifty version
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty
    except ImportError:
        import nifty_with_gurobi as nifty

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class NodesToBlocks(luigi.Task):

    pathToSeg = luigi.Parameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()
    dtype        = luigi.ListParameter('uint32')

    def requires(self):
        if PipelineParameter().defectPipeline:
            return {"seg": ExternalSegmentation(self.pathToSeg),
                    "defect_slices": DefectSliceDetection(self.pathToSeg)}
        else:
            return {"seg": ExternalSegmentation(self.pathToSeg)}

    @run_decorator
    def run(self):

        inp = self.input()
        seg = inp["seg"]
        seg.open()

        # if we have defects, we need to skip the completly defected slices in the node extraction,
        # because nodes inside them are completely excluded from the graph now
        if PipelineParameter().defectPipeline:
            defect_slices = vigra.readHDF5(inp["defect_slices"].path, 'defect_slices').astype('int64').tolist()
            workflow_logger.info("NodesToBlocks: Skipping slices %s due to defects." % str(defect_slices))
        else:
            defect_slices = []

        blocking = nifty.tools.blocking(roiBegin=[0L, 0L, 0L], roiEnd=seg.shape(), blockShape=self.blockShape)
        number_of_blocks = blocking.numberOfBlocks
        block_overlap = list(self.blockOverlap)

        n_workers = min(number_of_blocks, PipelineParameter().nThreads)
        # nWorkers = 1
        block_result = nifty.tools.nodesToBlocksStacked(seg.get(), blocking, block_overlap, defect_slices, n_workers)

        block_result = [np.array(b_res, dtype=self.dtype) for b_res in block_result]
        self.output().writeVlen(block_result)

    def output(self):
        block_string = '_'.join(map(str, self.blockShape))
        overlap_string = '_'.join(map(str, self.blockOverlap))
        save_name = "NodesToBlocks_%s_%s_%s.h5" % (
            block_string, overlap_string, "modified" if PipelineParameter().defectPipeline else "standard"
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# TODO modify s.t. they are also adjacent if they only touch via a common face
# constructs a graph from the given block shape and overlap
# in this graph, two blocks are adjacent if they have overlap
class BlockGridGraph(luigi.Task):

    pathToSeg = luigi.Parameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()

    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    @run_decorator
    def run(self):

        # get the shape
        inp = self.input()
        inp.open()
        shape = inp.shape()
        overlap = list(self.blockOverlap)

        # construct the blocking
        blocking = nifty.tools.blocking(
            roiBegin=[0L, 0L, 0L],
            roiEnd=shape,
            blockShape=list(map(long, self.blockShape))
        )
        n_blocks = blocking.numberOfBlocks

        # init the graphs
        # block graph connecting all blocks sharing overlap
        block_graph = nifty.graph.UndirectedGraph(n_blocks)
        # block graph only connecting nodes in 6 adjacency
        block_graph_nn = nifty.graph.UndirectedGraph(n_blocks)

        # construct the graph by iterating over the blocks and for each block finding blocks with overlap
        for block_id in xrange(n_blocks):
            block = blocking.getBlockWithHalo(block_id, overlap).outerBlock

            # find adjacent blocks via all blocks in the bounding box and excluding the current block id
            inner_block = blocking.getBlockWithHalo(block_id, overlap).innerBlock
            inner_begin, inner_end =  np.array(inner_block.begin), np.array(inner_block.end)
            center = (inner_end + inner_begin) / 2.

            adjacent_blocks = np.array(blocking.getBlockIdsOverlappingBoundingBox(block.begin, block.end, overlap))
            adjacent_blocks = adjacent_blocks[adjacent_blocks != block_id]

            assert adjacent_blocks.size, adjacent_blocks
            # constuct the edge vector for edges to all adjacent blocks
            block_edges = np.sort(
                np.concatenate(
                    [block_id * np.ones((len(adjacent_blocks), 1)), adjacent_blocks[:, None]],
                    axis=1
                ),
                axis=1
            ).astype('uint32')
            block_graph.insertEdges(block_edges)

            # add edges to the nearest neighbours
            for adj_id in adjacent_blocks:

                if adj_id == block_id:
                    continue

                adj_block = blocking.getBlock(adj_id)
                adj_begin, adj_end = np.array(adj_block.begin), np.array(adj_block.end)
                adj_center = (adj_end + adj_begin) / 2.

                block_dist = np.absolute(center - adj_center)

                # coordinates for nearest neighbors are only different in a single dimension
                if np.sum(block_dist > 0) == 1:
                    block_graph_nn.insertEdge(min(block_id, adj_id), max(block_id, adj_id))

        # serialize the graphs
        out = self.output()
        out.write(block_graph.serialize())
        out.write(block_graph_nn.serialize(), 'nearest_neighbors')

    def output(self):
        block_string = '_'.join(map(str, self.blockShape))
        overlap_string = '_'.join(map(str, self.blockOverlap))
        save_name = "BlockGridGraph_%s_%s.h5" % (block_string, overlap_string)
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# computes the edges between blocks for a given graph / problem
# based on the outer edges in this problem
class EdgesBetweenBlocks(luigi.Task):

    pathToSeg    = luigi.Parameter()
    problem      = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()
    level        = luigi.Parameter()

    # TODO try both combinations
    outerEdgesOnly = luigi.Parameter(default=True)

    def requires(self):
        return {
            'block_graph': BlockGridGraph(self.pathToSeg, self.blockShape, self.blockOverlap),
            'problem': self.problem,
            'nodes_to_blocks': NodesToBlocks(self.pathToSeg, self.blockShape, self.blockOverlap)
        }

    @run_decorator
    def run(self):

        # get the inputs:
        inp = self.input()

        # graph of the current problem
        graph = nifty.graph.UndirectedGraph()
        graph.deserialize(inp['problem'].read('graph'))

        # outer edges
        if self.outerEdgesOnly:
            uv_ids = graph.uvIds()
            outer_edges = inp['problem'].read('outer_edges')
            outer_uvs = uv_ids[outer_edges]

        # nodes to blocks and block_graph
        block_graph = nifty.graph.UndirectedGraph()
        block_graph.deserialize(inp['block_graph'].read())
        nodes_to_blocks = inp['nodes_to_blocks'].read()
        assert len(nodes_to_blocks) == block_graph.numberOfNodes

        # we need to project the node ids to the reduced node ids in the current level
        to_new_nodes = inp['problem'].read('global2new')
        nodes_to_blocks_new = [
            np.unique(to_new_nodes[nodes_to_blocks[block_id]]) for block_id in xrange(len(nodes_to_blocks))
        ]

        # TODO this seems to be quite expensive and could probably be parallelized
        # iterate over the adjacent blocks and save the outer edge-ids that connect the blocks
        edges_between_blocks = []
        for block_u, block_v in block_graph.uvIds():

            # get the nodes in both blocks
            nodes_u, nodes_v = nodes_to_blocks_new[block_u], nodes_to_blocks_new[block_v]

            # find the intersection of the nodes with the graph edges
            # by first constructing all potential edges (= all combinations of nodes in block u, v)
            # then searching these edges in the graph
            potential_edges = cartesian([nodes_u, nodes_v])
            if self.outerEdgesOnly:
                matches = find_matching_row_indices(outer_uvs, potential_edges)[:, 0]
                bb_edges = outer_edges[matches]
            else:
                matches = graph.findEdges(potential_edges)
                bb_edges = matches[matches != -1]

            edges_between_blocks.append(bb_edges)

        out = self.output()

        # serialize the between block edges
        out.writeVlen(edges_between_blocks, 'edges_between_blocks')
        # serialize the block uv ids for convinience
        out.write(block_graph.uvIds(), 'block_uv_ids')

    def output(self):
        save_name = "EdgesBetweenBlocks_L%i_%s_%s.h5" % (
            self.level,
            '_'.join(map(str, self.blockShape)),
            '_'.join(map(str, self.blockOverlap))
        )
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)
