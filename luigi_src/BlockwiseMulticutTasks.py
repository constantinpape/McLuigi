# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

from PipelineParameter import PipelineParameter
from DataTasks import RegionAdjacencyGraph, ExternalSegmentationLabeled
from MulticutSolverTasks import MCProblem#, MCSSolverOpengmFusionMoves, MCSSolverOpengmExact

from Tools import UnionFind

import logging
import json

import numpy as np
import vigra
from concurrent import futures

# TODO would be better to let the scheduler handle the parallelisation
# -> start corresponding MCProblem / MCSolver subtasks in parallel
# once we know how this works...

# so this is only preliminary
# -> need more tasks and more atomicity!

# TODO use Fusionmoves task instead
def fusion_moves(uv_ids, edge_costs, id):

    # read the mc parameter
    with open(PipelineParameter().MCConfigFile, 'r') as f:
        MCConfig = json.load(f)

    n_var = uv_ids.max() + 1

    # set up the opengm model
    states = np.ones(n_var) * n_var
    gm = opengm.gm(states)

    # potts model
    potts_shape = [n_var, n_var]

    potts = opengm.pottsFunctions(potts_shape, np.zeros_like( edge_costs ), edge_costs )

    # potts model to opengm function
    fids_b = gm.addFunctions(potts)

    gm.addFactors(fids_b, uv_ids)

    pparam = opengm.InfParam(seedFraction = MCConfig["SeedFraction"])
    parameter = opengm.InfParam(generator = 'randomizedWatershed',
                                proposalParam = pparam,
                                numStopIt = MCConfig["NumItStop"],
                                numIt = MCConfig["NumIt"])

    inf = opengm.inference.IntersectionBased(gm, parameter=parameter)

    t_inf = time.time()
    inf.infer()
    t_inf = time.time() - t_inf

    logging.info("Inference for subblock " + str(id) + " with fusion moves solver in " + str(t_inf) + " s")

    res_node = inf.arg()

    return res_node


class BlockwiseMulticutSolver(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return {"SubSolutions" : BlockwiseSubSolver( self.PathToSeg, self.PathToRF ),
                "GlobalProblem" : MCProblem( self.PathToSeg, self.PathToRF) }

    def run(self):

        problem = self.input()["GlobalProblem"].read()
        CutEdges = self.input()["SubSolutions"].read()

        uvIds = problem[:,:2]
        costs = problem[:,2]

        n_nodes = uvIds.max() + 1

        # set up the opengm model
        states = np.ones(n_nodes) * n_nodes
        gm_global = opengm.gm(states)

        potts_shape = [n_nodes, n_nodes]
        potts = opengm.pottsFunctions(potts_shape, np.zeros_like( costs ), costs )

        # potts model to opengm function
        fids_b = gm_global.addFunctions(potts)

        gm_global.addFactors(fids_b, uvIds)

        # merge nodes according to cut edges
        # this means, that we merge all segments that have an edge with value 0 in between
        # for this, we use a ufd datastructure
        udf = UnionFind( n_nodes )

        MergeNodes = uvIds[CutEdges == 0]

        for pair in MergeNodes:
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
        active_edges = np.where( CutEdges == 1 )[0]
        new_edges_dict = {}
        for edge_id in active_edges:
            u_old = uvIds[edge_id][0]
            v_old = uvIds[edge_id][1]
            n_0_new = old_to_new_nodes[u_old]
            n_1_new = old_to_new_nodes[v_old]
            # we have to bew in different new nodes!
            assert n_0_new != n_1_new, str(n_0_new) + " , " + str(n_1_new)
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

        logging.info("Merging of blockwise results reduced problemsize:" )
        logging.info("Nodes: From" + n_nodes + "to" + n_nodes_new )
        logging.info("Edges: From" + n_edges + "to" + n_edges_new )

        res_node_new = fusion_moves( uv_ids_new, costs_new )

        assert res_node_new.shape[0] == n_nodes_new

        # project back to old problem
        res_node = np.zeros(n_nodes, dtype = np.uint32)
        for n_id in xrange(res_node_new.shape[0]):
            for old_node in new_to_old_nodes[n_id]:
                res_node[old_node] = res_node_new[n_id]

        # get the global energy
        E_glob = gm_global.evaluate(res_node)
        logging.info()

        if 0 in res_node:
            res_node += 1

        self.output().write(res_node)


        def output(self):
            save_path = os.path.join( PipelineParameter().cache, "MCBlockwise.h5")
            return HDF5Target( save_path )


class BlockwiseSubSolver(luigi.Task):

    PathToSeg = luigi.Parameter()
    PathToRF  = luigi.Parameter()

    def requires(self):
        return {"RAG" : RegionAdjacencyGraph(self.PathToSeg),
                "Seg" : ExternalSegmentationLabeled(self.PathToSeg),
                "GlobalProblem" : MCProblem( self.PathToSeg, self.PathToRF) }

    def run(self):

        # Input
        rag = self.input()["RAG"].read()
        seg = self.input()["Seg"].read()
        problem = self.input()["GlobalProblem"].read()
        costs = problem[:,2]

        # TODO make this a task and make this more efficient
        def extract_subproblems(seg_local, rag_global):

            # get the nodes in this blocks
            nodes = np.unique(seg_local)
            # global nodes to local nodes
            global_to_local_nodes = {}
            for i in xrange(nodes.shape[0]):
                global_to_local_nodes[nodes[i]] = i

            # get the edges and uvids in this block
            inner_edges  = []
            outer_edges  = []
            uv_ids_local = {}
            for n_id in nodes:
                node = rag_global.nodeFromId( long(n_id) )
                for adj_node in rag_global.neighbourNodeIter(node):
                    edge = rag_global.findEdge(node,adj_node)
                    if adj_node.id in nodes:
                        inner_edges.append(edge.id)
                        u_local = global_to_local_nodes[n_id]
                        v_local = global_to_local_nodes[adj_node.id]
                        uv_ids_local[edge.id] = [ u_local, v_local ]
                    else:
                        outer_edges.append(edge.id)

            # need to get rid of potential duplicates and order uv-ids propperly
            inner_edges = np.unique(inner_edges)
            outer_edges = np.unique(outer_edges)
            uv_ids      = np.zeros( (inner_edges.shape[0], 2), dtype = np.uint32 )
            for i in xrange( inner_edges.shape[0] ):
                edge_id   = inner_edges[i]
                uv_ids[i] = uv_ids_local[edge_id]
            uv_ids = np.sort(uv_ids, axis = 1)

            assert uv_ids.max() == nodes.shape[0] - 1, str(uv_ids.max()) + " , " +str(nodes.shape[0] - 1)

            return (inner_edges, outer_edges, uv_ids)


        # block parameters
        with open(PipelineParameter().MCConfigFile, 'r') as f:
            MCConfig = json.load(f)

        BlockSize = MCConfig["BlockSize"]
        BlockOverlap = MCConfig["BlockOverlap"]

        # calculate number and locations of the subblocks
        Shape = seg.shape

        sX = BlockSize[0]
        sY = BlockSize[1]
        sZ = BlockSize[2]

        oX = BlockOverlap[0]
        oY = BlockOverlap[1]
        oZ = BlockOverlap[2]

        nX = int( np.ceil( float( sX / Shape[0] ) ) )
        nY = int( np.ceil( float( sY / Shape[1] ) ) )
        nZ = int( np.ceil( float( sZ / Shape[2] ) ) )

        nBlocks = nX * nY * nZ

        logging.info("Fitting " + str(nBlocks) + " blocks of size " + str(BlockSize) + " into shape " + str(Shape))

        slicings = []
        for x in xrange(nX):

            # X range
            startX = x * sX
            if x != 0:
                startX -= oX
            endX = (x + 1) * sX + oX
            if endX > Shape[0]:
                x = Shape[0]

            for y in xrange(nY):

                # Y range
                startY = y * sY
                if y != 0:
                    startY -= oY
                endY = (y + 1) * sY + oY
                if endY > Shape[1]:
                    y = Shape[1]

                for z in xrange(nZ):

                    # Z range
                    startZ = y * sZ
                    if z != 0:
                        startZ -= oZ
                    endZ = (z + 1) * sZ + oz
                    if endZ > Shape[2]:
                        z = Shape[2]

                    slicings.append( np.s_[startX:endX,startY:endY,startZ:endZ] )

        #nWorkers = 4
        nWorkers = min( nBlocks, PipelineParameter().n_threads )
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for id, slicing in enumerate(slicings):
                logging.info( "Block id " + str(id) + " slicing " + str(slicing) )
                tasks.append( executor.submit( seg[slicing], rag  ) )

        SubProblems = [task.result() for task in tasks]

        t_inf_total = time.time()

        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            for id, SubProblem in enumerate(SubProblems):
                tasks.append( executor.submit( fusion_moves, SubProblem[2],
                    costs[SubProblem[0]], id ) )

        SubResults = [task.result() for task in tasks]

        t_inf_total = t_inf_total - time.time()
        log.info( "Inference time total for subproblems " + str(t_inf_total)  + " s")

        nEdges = rag.edgeNum
        CutEdges = np.zeros( nEdges, dtype = np.uint8 )

        assert len(SubResults) == len(SubProblems)

        for id in xrange(nBlocks):

            # get the cut edges from the subproblem
            nodeRes = SubResults[id]
            uvIdsSub = SubProblem[id][2]

            ru = nodeRes[uvIdsSub[:,0]]
            rv = nodeRes[uvIdsSub[:,1]]
            edgeRes = ru!=rv

            # add up cut inner edges
            CutEdges[SubProblem[id][0]] += edgeRes

            # add up outer edges
            CutEdges[SubProblem[id][1]] += 1

        # all edges which are cut at least once will be cut
        CutEdges[cut_edges >= 1] = 1

        self.output().write(CutEdges)


    def output(self):
        return HDF5Target(os.path.join( PipelineParameter().cache, "BlockwiseSubSolver.h5" ) )


