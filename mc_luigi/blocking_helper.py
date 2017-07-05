import luigi
from tools import run_decorator, config_logger, cartesian
from customTargets import HDF5DataTarget
from dataTasks import ExternalSegmentation
from pipelineParameter import PipelineParameter
from defectDetectionTasks import DefectSliceDetection

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
        save_name = "NodesToBlocks_%s_%s_%s.h5" % (block_string, overlap_string, "modified" if PipelineParameter().defectPipeline else "standard")
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

        # init the graph
        block_graph = nifty.graph.UndirectedGraph(n_blocks)
        print overlap

        # construct the graph by iterating over the blocks and for each block finding blocks with overlap
        for block_id in xrange(n_blocks):
            block = blocking.getBlockWithHalo(block_id, overlap).outerBlock

            # find adjacent blocks via all blocks in the bounding box and excluding the current block id
            adjacent_blocks = np.array(blocking.getBlockIdsOverlappingBoundingBox(block.begin, block.end, overlap))
            adjacent_blocks = adjacent_blocks[adjacent_blocks != block_id]

            assert adjacent_blocks.size, adjacent_blocks
            # construct edge vector and append it
            block_edges = np.sort(
                np.concatenate(
                    [block_id * np.ones((len(adjacent_blocks), 1)), adjacent_blocks[:, None]],
                    axis=1
                ),
                axis=1
            ).astype('uint32')

            block_graph.insertEdges(block_edges)

        # serialize the graph
        out = self.output()
        out.write(block_graph.serialize())


    def output(self):
        block_string = '_'.join(map(str, self.blockShape))
        overlap_string = '_'.join(map(str, self.blockOverlap))
        save_name = "BlockGridGraph_%s_%s.h5" % (block_string, overlap_string)
        save_path = os.path.join(PipelineParameter().cache, save_name)
        return HDF5DataTarget(save_path)


# NOTE: This is not done in the most efficient manner:
# we extract the global (== original segmentation) nodes for the current level / problem
# again and also construct the block graph for the current level again (although the latter is not a performance issue)
# instead one could use the nodes extracted via 'NodesToBlocks' from the initial blocking (level 1) and then map to
# the current blocking.
# However, this is more complicated in terms of mapping between different blockings, that's why it is not implemented this way
# for now. If this is ever put into production for larger volumes, this should be changed.

# computes the edges between blocks for a given graph / problem
class EdgesBetweenBlocks(luigi.Task):

    pathToSeg    = luigi.Parameter()
    problem      = luigi.TaskParameter()

    blockShape   = luigi.ListParameter()
    blockOverlap = luigi.ListParameter()
    level        = luigi.Parameter()

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
        # iterate over the adjacent blocks and save all graph edges that connect adjacent blocks
        edges_between_blocks = []
        for block_u, block_v in block_graph.uvIds():

            # get the nodes in both blocks
            nodes_u, nodes_v = nodes_to_blocks_new[block_u], nodes_to_blocks_new[block_v]

            # find the intersection of the nodes with the graph edges
            # by first constructing all potential edges (= all combinations of nodes in block u, v)
            # then searching these edges in the graph
            potential_edges = cartesian([nodes_u, nodes_v])
            edge_ids = graph.findEdges(potential_edges)
            edge_ids = edge_ids[edge_ids != -1]  # exclude all invalid edge ids
            edges_between_blocks.append(edge_ids)

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
