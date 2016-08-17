# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from multicutSolverTasks import McProblem#, McSolverFusionMoves, MCSSolverOpengmExact
from customTargets import HDF5DataTarget

from toolsLuigi import UnionFind, config_logger

import os
import logging
import json
import time

import numpy as np
import vigra
import nifty

from concurrent import futures

# TODO would be better to let the scheduler handle the parallelisation
# -> start corresponding MCProblem / MCSolver subtasks in parallel
# once we know how this works...

# so this is only preliminary
# -> need more tasks and more atomicity!

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# TODO use Fusionmoves task instead
def fusion_moves(uv_ids, edge_costs, id, n_parallel):

    # read the mc parameter
    # TODO actually use these
    with open(PipelineParameter().MCConfigFile, 'r') as f:
        mc_config = json.load(f)

    n_var = uv_ids.max() + 1

    g = nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_costs.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_costs)

    workflow_logger.info("Solving MC Problem with " + str(n_var) + " number of variables")
    workflow_logger.info("Using nifty fusionmoves")

    greedy = obj.greedyAdditiveFactory().create(obj)

    ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    )

    solver = obj.fusionMoveBasedFactory(
        verbose=mc_config["verbose"],
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=mc_config["sigmaFusion"],seedFraction=mc_config["seedFraction"]),
        numberOfIterations=mc_config["numIt"],
        numberOfParallelProposals=n_parallel*2, # we use the parameter n parallel instead of the config here
        numberOfThreads=n_parallel,
        stopIfNoImprovement=mc_config["numItStop"],
        fuseN=mc_config["numFuse"],
    ).create(obj)

    t_inf = time.time()
    ret = greedy.optimize()
    ret = solver.optimize(nodeLabels=ret)
    t_inf = time.time() - t_inf

    workflow_logger.info("Inference for block " + str(id) + " with fusion moves solver in " + str(t_inf) + " s")

    return ret


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        globalProblem = McProblem( self.PathToSeg, self.PathToRF)
        return {"SubSolutions" : BlockwiseSubSolver( self.PathToSeg, globalProblem ),
                "GlobalProblem" : globalProblem }

    def run(self):

        inp = self.input()
        problem   = inp["GlobalProblem"].read()
        cut_edges = inp["SubSolutions"].read()

        uv_ids = problem[:,:2].astype(np.uint32)
        costs  = problem[:,2]

        n_nodes = uv_ids.max() + 1

        # set up the global problem to later evaluate the energies
        g = nifty.graph.UndirectedGraph(int(n_nodes))
        g.insertEdges(uv_ids)
        global_obj = nifty.graph.multicut.multicutObjective(g, costs)

        # TODO use nifty datastructure and continue when subproblems are adapted
        udf = UnionFind( n_nodes )

        merge_nodes = uv_ids[cut_edges == 0]

        for pair in merge_nodes:
            u = pair[0]
            v = pair[1]
            udf.merge(u, v)

        # we need to get the result of the merging
        new_to_old_nodes = udf.get_merge_result()
        # number of nodes for the new problem
        n_nodes_new = len(new_to_old_nodes)

        # find old to new nodes
        old_to_new_nodes = np.zeros( n_nodes, dtype = np.uint32 )
        for set_id in xrange( n_nodes_new ):
            for n_id in new_to_old_nodes[set_id]:
                assert n_id < n_nodes, str(n_id) + " , " + str(n_nodes)
                old_to_new_nodes[n_id] = set_id

        # find new edges and new edge weights
        active_edges = np.where( cut_edges == 1 )[0]
        new_edges_dict = {}
        for edge_id in active_edges:
            u_old = uv_ids[edge_id,0]
            v_old = uv_ids[edge_id,1]
            n_0_new = old_to_new_nodes[u_old]
            n_1_new = old_to_new_nodes[v_old]
            # we have to be in different new nodes! FIXME something is going wrong here along the way
            assert n_0_new != n_1_new, str(n_0_new) + " , " + str(n_1_new) + " @ edge: " + str(edge_id)
            # need to order to always have the same keys
            u_new = min(n_0_new, n_1_new)
            v_new = max(n_0_new, n_1_new)
            # need to check if have already come by this new edge
            if (u_new,v_new) in new_edges_dict:
                new_edges_dict[(u_new,v_new)] += costs[edge_id]
            else:
                new_edges_dict[(u_new,v_new)]  = costs[edge_id]

        n_edges_new = len( new_edges_dict.keys() )
        uv_ids_new = np.array( new_edges_dict.keys() )
        assert uv_ids_new.shape[0] == n_edges_new, str(uv_ids_new.shape[0]) + " , " + str(n_edges_new)
        # this should have the correct order
        costs_new = np.array( new_edges_dict.values() )

        workflow_logger.info("Merging of blockwise results reduced problemsize:" )
        workflow_logger.info("Nodes: From " + str(n_nodes) + " to " + str(n_nodes_new) )
        workflow_logger.info("Edges: From " + str(uv_ids.shape[0]) + " to " + str(n_edges_new) )

        res_node_new = fusion_moves( uv_ids_new, costs_new, "reduced global", 20 )

        assert res_node_new.shape[0] == n_nodes_new

        # project back to old problem
        res_node = np.zeros(n_nodes, dtype = np.uint32)
        for n_id in xrange(res_node_new.shape[0]):
            for old_node in new_to_old_nodes[n_id]:
                res_node[old_node] = res_node_new[n_id]

        # get the global energy
        e_glob = global_obj.evalNodeLabels(res_node)
        workflow_logger.info("Blockwise Multicut problem solved with energy " + str(e_glob) )

        self.output().write(res_node)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSolver.h5")
        return HDF5DataTarget( save_path )


class BlockwiseSubSolver(luigi.Task):

    PathToSeg = luigi.Parameter()
    globalProblem  = luigi.TaskParameter()

    def requires(self):
        return {"Rag" : StackedRegionAdjacencyGraph(self.PathToSeg),
                "Seg" : ExternalSegmentation(self.PathToSeg),
                "GlobalProblem" : self.globalProblem }

    def run(self):

        # Input
        inp = self.input()
        rag = inp["Rag"].read()
        seg = inp["Seg"]
        seg.open()
        problem = inp["GlobalProblem"].read()
        costs = problem[:,2]

        def extract_subproblems(seg_global, rag_global, block_begin, block_end):
            node_list = np.unique( seg_global.read(block_begin, block_end) )
            return nifty.graph.rag.extractNodesAndEdgesFromNodeList(rag_global, node_list)


        # block parameters
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            mc_config = json.load(f)

        block_size    = mc_config["blockSize"]
        block_overlap = mc_config["blockOverlap"]

        # TODO encapsulate and possibly use some vigra / nifty function here!
        s_z = block_size[0]
        assert s_z < seg.shape[0], str(s_z) + " , " + str(seg.shape[0])
        s_y = block_size[1]
        assert s_y < seg.shape[1], str(s_y) + " , " + str(seg.shape[1])
        s_x = block_size[2]
        assert s_x < seg.shape[2], str(s_x) + " , " + str(seg.shape[2])

        o_z = block_overlap[0]
        o_y = block_overlap[1]
        o_x = block_overlap[2]

        n_z = int( np.ceil( float( seg.shape[0] ) / s_z ) )
        n_y = int( np.ceil( float( seg.shape[1] ) / s_y ) )
        n_x = int( np.ceil( float( seg.shape[2] ) / s_x ) )

        n_blocks = n_x * n_y * n_z

        workflow_logger.info("Fitting " + str(n_blocks) + " blocks of size " + str(block_size) + " into shape " + str(seg.shape) + " additional overlaps: " + str(block_overlap))

        block_begins = []
        block_ends   = []
        for z in xrange(n_z):

            # z range
            start_z = z * s_z
            if z != 0:
                start_z -= o_z
            end_z = (z + 1) * s_z + o_z
            if end_z > seg.shape[0]:
                end_z = seg.shape[0]

            for y in xrange(n_y):

                # Y range
                start_y = y * s_y
                if y != 0:
                    start_y -= o_y
                end_y = (y + 1) * s_y + o_y
                if end_y > seg.shape[1]:
                    end_y = seg.shape[1]

                for x in xrange(n_x):

                    # x range
                    start_x = x * s_x
                    if x != 0:
                        start_x -= o_x
                    end_x = (x + 1) * s_x + o_x
                    if end_x > seg.shape[2]:
                        end_x = seg.shape[2]

                    block_begins.append( [start_z,start_y,start_x] )
                    block_ends.append(   [end_z,end_y,end_x] )

        #nWorkers = 1
        nWorkers = min( n_blocks, PipelineParameter().nThreads )

        t_extract = time.time()

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for block_id in xrange(len(block_begins)):
                workflow_logger.info( "Block id " + str(block_id) + " start " + str(block_begins[block_id]) + " end " + str(block_ends[block_id]) )
                tasks.append( executor.submit( extract_subproblems, seg, rag, block_begins[block_id], block_ends[block_id]  ) )

        sub_problems = [task.result() for task in tasks]

        assert len(sub_problems) == n_blocks, str(len(sub_problems)) + " , " + str(n_blocks)

        t_extract = time.time() - t_extract
        workflow_logger.info( "Extraction time for subproblems " + str(t_extract)  + " s")

        t_inf_total = time.time()

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for id, sub_problem in enumerate(sub_problems):
                tasks.append( executor.submit( fusion_moves, sub_problem[2],
                    costs[sub_problem[0]], id, 1 ) )
        sub_results = [task.result() for task in tasks]

        #sub_results = []
        #for id, sub_problem in enumerate(sub_problems):
        #    sub_results.append( fusion_moves(sub_problem[2], costs[sub_problem[0]], id) )

        t_inf_total = time.time() - t_inf_total
        workflow_logger.info( "Inference time total for subproblems " + str(t_inf_total)  + " s")

        n_edges = rag.numberOfEdges
        cut_edges = np.zeros( n_edges, dtype = np.uint8 )

        assert len(sub_results) == len(sub_problems), str(len(sub_results)) + " , " + str(len(sub_problems))

        for id in xrange(n_blocks):

            # get the cut edges from the subproblem
            node_res = sub_results[id]
            uv_ids_sub = sub_problems[id][2]

            ru = node_res[uv_ids_sub[:,0]]
            rv = node_res[uv_ids_sub[:,1]]
            edge_res = ru!=rv

            # add up cut inner edges
            cut_edges[sub_problems[id][0]] += edge_res

            # add up outer edges
            cut_edges[sub_problems[id][1]] += 1

        # all edges which are cut at least once will be cut
        cut_edges[cut_edges >= 1] = 1

        self.output().write(cut_edges)


    def output(self):
        return HDF5DataTarget(os.path.join( PipelineParameter().cache, "BlockwiseSubSolver.h5" ) )


