# Multicut Pipeline implemented with luigi
# Blockwise solver tasks

import luigi

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation
from multicutSolverTasks import McProblem#, McSolverFusionMoves, MCSSolverOpengmExact
from customTargets import HDF5DataTarget

from toolsLuigi import UnionFind, config_logger, get_blocks

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


# serialization and deserialization for list of lists with hdf5

def serializeNew2Old(out, new2old):
    # we can't serialize the whole list of list, so we do it in this ugly manner...
    for i, nList in enumerate(new2old):
        out.write(nList, "new2old/" + str(i))


def deserializeNew2Old(inp):
    new2old = []
    nodeIndex = 0
    while True:
        try:
            new2old.append( inp.read("new2old/" + str(nodeIndex) ) )
            nodeIndex += 1
        except KeyError:
            break
    return new2old



# TODO use Fusionmoves task instead
def fusion_moves(uv_ids, edge_costs, id, n_parallel):

    # read the mc parameter
    with open(PipelineParameter().MCConfigFile, 'r') as f:
        mc_config = json.load(f)

    n_var = uv_ids.max() + 1

    g = nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_costs.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_costs)

    workflow_logger.debug("Solving MC Problem with " + str(n_var) + " number of variables")
    workflow_logger.debug("Using nifty fusionmoves")

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

    workflow_logger.debug("Inference for block " + str(id) + " with fusion moves solver in " + str(t_inf) + " s")

    return ret


# produce reduced global graph from subproblems
# solve global multicut problem on the reduced graph
class BlockwiseMulticutSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToRF  = luigi.Parameter()

    numberOflevels = luigi.Parameter()
    #initialBlocksize = luigi.Parameter()

    def requires(self):
        globalProblem = McProblem( self.pathToSeg, self.pathToRF)

        # for now, we hardcode the block overlaps here and don't change it throughout the different hierarchy levels.
        # in the end, this sould also be a parameter (and maybe even change in the different hierarchy levels)
        blockOverlap = [4,20,20]
        # nested block hierarchy TODO make it possible to set this more explicit.

        # 1st hierarchy level: 50 x 512 x 512 blocks (hardcoded for now)
        initialBlockSize = [50,512,512]

        problemHierarchy = [globalProblem,]
        blockFactor = 1
        for l in xrange(self.numberOflevels):
            levelBlocksize = map( lambda x: x*blockFactor, initialBlockSize)
            workflow_logger.info("Scheduling reduced problem for level " + str(l) + " with block size: " + str(levelBlocksize))
            # TODO check that we don't get larger than the actual shape here
            problemHierarchy.append( ReducedProblem(self.pathToSeg, problemHierarchy[-1],
                levelBlocksize , blockOverlap, l) )
            blockFactor *= 2

        return problemHierarchy


    def run(self):

        problems = self.input()

        globalProblem = problems[0]

        graphGlobal = nifty.graph.UndirectedGraph()
        graphGlobal.deserialize(globalProblem.read("graph"))

        uv_ids_glob = graphGlobal.uvIds()
        costs_glob  = globalProblem.read("costs")

        n_nodes_glob = uv_ids_glob.max() + 1

        global_obj = nifty.graph.multicut.multicutObjective(graphGlobal, costs_glob)

        # we solve the problem for the costs and the edges of the final hierarchy level
        lastProblem = problems[-1]
        reducedGraph = nifty.graph.UndirectedGraph()
        reducedGraph.deserialize( lastProblem.read("graph") )

        uv_ids_last = reducedGraph.uvIds()
        costs_last = lastProblem.read("costs")

        new2old_last = deserializeNew2Old(lastProblem)
        assert len(new2old_last) == uv_ids_last.max() + 1, str(len(new2old_last)) + " , " + str(uv_ids_last.max() + 1)

        t_inf = time.time()
        # FIXME parallelism makes it slower here -> investigate this further and discuss with thorsten!
        #res_node_new = fusion_moves( uv_ids_new, costs_new, "reduced global", 20 )
        res_node_last = fusion_moves( uv_ids_last, costs_last, "reduced global", 1 )
        t_inf = time.time() - t_inf
        workflow_logger.info("Inference of reduced problem for the whole volume took: " + str(t_inf) + " s")

        assert res_node_last.shape[0] == len(new2old_last)

        # project back to global problem through all hierarchy levels
        resNode = res_node_last
        problem = lastProblem
        new2old = new2old_last
        for l in reversed(xrange(self.numberOflevels)):

            nextProblem = problems[l]
            if l != 0:
                nextNew2old = deserializeNew2Old( nextProblem )
                nextNumberOfNodes = len( nextNew2old )
            else:
                nextNumberOfNodes = n_nodes_glob

            nextResNode = np.zeros(nextNumberOfNodes, dtype = 'uint32')
            for n_id in xrange(resNode.shape[0]):
                for old_node in new2old[n_id]:
                    nextResNode[old_node] = resNode[n_id]

            resNode = nextResNode
            if l != 0:
                new2old = nextNew2old

        # get the global energy
        e_glob = global_obj.evalNodeLabels(resNode)
        workflow_logger.info("Blockwise Multicut problem solved with energy " + str(e_glob) )

        self.output().write(resNode)


    def output(self):
        save_path = os.path.join( PipelineParameter().cache, "BlockwiseMulticutSolver.h5")
        return HDF5DataTarget( save_path )


class ReducedProblem(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockSize     = luigi.ListParameter()
    blockOverlaps = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return {"subSolution" : BlockwiseSubSolver( self.pathToSeg, self.problem, self.blockSize, self.blockOverlaps, self.level ),
                "problem" : self.problem }


    def run(self):

        inp = self.input()
        problem   = inp["problem"]
        cut_edges = inp["subSolution"].read()

        g = nifty.graph.UndirectedGraph()
        g.deserialize(problem.read("graph"))

        uv_ids = g.uvIds().astype('uint32')
        n_nodes = uv_ids.max() + 1
        costs  = problem.read("costs")

        # TODO use something faster, maybe expose nifty ufd to python
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
            # we have to be in different new nodes!
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

        workflow_logger.info("Merging of blockwise results reduced problemsize:" )
        workflow_logger.info("Nodes: From " + str(n_nodes) + " to " + str(n_nodes_new) )
        workflow_logger.info("Edges: From " + str(uv_ids.shape[0]) + " to " + str(n_edges_new) )

        uv_ids_new = np.array( new_edges_dict.keys() )

        reducedGraph = nifty.graph.UndirectedGraph(n_nodes_new)
        reducedGraph.insertEdges(uv_ids_new)

        assert uv_ids_new.shape[0] == n_edges_new, str(uv_ids_new.shape[0]) + " , " + str(n_edges_new)
        # this should have the correct order
        costs_new = np.array( new_edges_dict.values() )

        if self.level == 0:
            global2new = old_to_new_nodes

        else:
            global2newLast = problem.read("global2new")
            global2new = np.zeros_like( global2newLast, dtype = np.uint32)
            for newNode in xrange(n_nodes_new):
                for oldNode in new_to_old_nodes[newNode]:
                    global2new[global2newLast[oldNode]] = newNode


        out = self.output()
        out.write(reducedGraph.serialize(), "graph")
        out.write(costs_new, "costs")
        out.write(global2new, "global2new")
        # need to serialize this differently, because hdf5 can't natively save lists of lists
        serializeNew2Old(out, new_to_old_nodes)


    def output(self):
        blcksize_str = "_".join( map( str, list(self.blockSize) ) )
        save_name = "ReducedProblem_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )



class BlockwiseSubSolver(luigi.Task):

    pathToSeg = luigi.Parameter()
    problem   = luigi.TaskParameter()

    blockSize     = luigi.ListParameter()
    blockOverlaps = luigi.ListParameter()

    level = luigi.Parameter()

    def requires(self):
        return { "seg" : ExternalSegmentation(self.pathToSeg), "problem" : self.problem }

    def run(self):

        # abusing python non-typedness...

        def extract_subproblems(seg_global, graph_global, block_begin, block_end, l, global2new):
            node_list = np.unique( seg_global.read(block_begin, block_end) )
            if l != 0:
                # TODO no for loop !
                for i in xrange(node_list.shape[0]):
                    node_list[i] = global2new[node_list[i]]
            return nifty.graph.extractSubgraphFromNodes(graph_global, node_list)

        # Input
        inp = self.input()
        seg = inp["seg"]
        seg.open()

        problem = inp["problem"]
        costs = problem.read("costs")

        graph = nifty.graph.UndirectedGraph()
        graph.deserialize( problem.read("graph") )
        n_edges = graph.numberOfEdges

        if self.level == 0:
            global2newNodes = None
        else:
            global2newNodes = problem.read("global2new")

        # TODO this function is implemented VERY ugly, best to replace this with vigra or nifty functionality
        n_blocks, block_begins, block_ends = get_blocks(seg.shape, self.blockSize, self.blockOverlaps)

        #nWorkers = 1
        nWorkers = min( n_blocks, PipelineParameter().nThreads )

        t_extract = time.time()
        #extract_subproblems( seg, graph, block_begins[0], block_ends[0], self.level, global2newNodes )
        #quit()

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for block_id in xrange(len(block_begins)):
                workflow_logger.debug( "Block id " + str(block_id) + " start " + str(block_begins[block_id]) + " end " + str(block_ends[block_id]) )
                tasks.append( executor.submit( extract_subproblems, seg, graph, block_begins[block_id], block_ends[block_id], self.level, global2newNodes ) )

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

        t_inf_total = time.time() - t_inf_total
        workflow_logger.info( "Inference time total for subproblems " + str(t_inf_total)  + " s")

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
        blcksize_str = "_".join( map( str, list(self.blockSize) ) )
        save_name = "BlockwiseSubSolver_" + blcksize_str + ".h5"
        save_path = os.path.join( PipelineParameter().cache, save_name)
        return HDF5DataTarget( save_path )
