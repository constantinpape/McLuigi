# Multicut Pipeline implemented with luigi
# Taksks for defect and handling
import luigi

from customTargets import HDF5DataTarget, HDF5VolumeTarget
from dataTasks import InputData, ExternalSegmentation, StackedRegionAdjacencyGraph
from defectDetectionTasks import DefectSliceDetection
from featureTasks import EdgeFeatures, RegionFeatures, RegionNodeFeatures

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json

import os
import time
import numpy as np
import vigra
import nifty

from concurrent import futures


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# map defect patches to overlapping superpixels
class DefectsToNodes(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'seg' : ExternalSegmentation(self.pathToSeg),
                'defects' : DefectSliceDetection(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        seg = inp['seg']
        seg.open()
        defects = inp['defects']
        defects.open()

        assert seg.shape == defects.shape

        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        def defects_to_nodes_z(z):
            slice_begin = [long(z),0L,0L]
            slice_end   = [long(z+1),ny,nx]
            defect_mask = defects.read(slice_begin,slice_end)
            if 1 in defect_mask:
                print z
                seg_z = seg.read(slice_begin,slice_end)
                where_defect = defect_mask == 1
                defect_nodes_slice = np.unique(seg_z[where_defect])
                return list(defect_nodes_slice), len(defect_nodes_slice) * [z]
            else:
                return [], []

        # non-parallel for debugging
        #defect_nodes = []
        #nodes_z = []
        #for z in xrange(seg.shape[0]):
        #    res = defects_to_nodes_z(z)
        #    defect_nodes.extend(res[0])
        #    nodes_z.extend(res[1])

        with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
            tasks = []
            for z in xrange(seg.shape[0]):
                tasks.append(executor.submit(defects_to_nodes_z,z))
            defect_nodes = []
            nodes_z      = []
            for fut in tasks:
                nodes, zz = fut.result()
                if nodes:
                    defect_nodes.extend(nodes)
                    nodes_z.extend(zz)

        assert len(defect_nodes) == len(nodes_z)

        workflow_logger.info("DefectsToNodes: found %i defected nodes" % (len(defect_nodes)))

        self.output().write(np.array(defect_nodes,dtype='uint32'),'defect_nodes')
        self.output().write(np.array(nodes_z,dtype='uint32'),'nodes_z')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "DefectsToNodes_%s_nBins%i_binThreshold%i.h5" % (segFile,
            PipelineParameter().nBinsSliceStatistics,
            PipelineParameter().binThreshold) )
        return HDF5DataTarget(save_path)


# modify the rag to isolate defected superpixels
# introduce skip edges over the defected superpixels in z
class ModifiedAdjacency(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'rag' : StackedRegionAdjacencyGraph(self.pathToSeg),
                'seg' : ExternalSegmentation(self.pathToSeg),
                'defect_nodes' : DefectsToNodes(self.pathToSeg)}

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

        edge_offset = rag.totalNumberOfInSliceEdges
        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        # loop over the defect nodes and find the skip edges,
        # compute the modify adjacency

        delete_edges = [] # the z-edges between defected and non-defected nodes that are deleted from the graph
        ignore_edges = [] # the xy-edges between defected and non-defected nodes, that will be set to maximally repulsive weights

        skip_edges   = [] # the skip edges that run over the defects in z
        skip_ranges  = [] # z-distance of the skip edges
        skip_starts  = [] # starting slices of the skip edges

        # get the original rag adjacency
        uv_ids = rag.uvIds()
        n_nodes = uv_ids.max() + 1

        defect_slices = np.unique(nodes_z)

        defect_node_dict = {int(z) : list(defect_nodes[nodes_z == z].astype(int)) for z in defect_slices}

        # FIXME TODO can't do this here once we have individual defect patches
        consecutive_defect_slices = np.split(defect_slices, np.where(np.diff(defect_slices) != 1)[0] + 1)
        has_lower_defect_list = []
        for consec in consecutive_defect_slices:
            if len(consec) > 1:
                has_lower_defect_list.extend(consec[1:])
        print "Slices with lower defect slice:", has_lower_defect_list

        def get_skip_edges_from_nifty(z, i):
            has_lower_defect = z in has_lower_defect_list
            delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = nifty.graph.rag.getSkipEdgesForSlice(
                rag,
                int(z),
                defect_node_dict,
                has_lower_defect)
            print 'Finished processing slice', z, ':', i, '/', len(defect_slices)
            return z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z

        # non-parallel for debugging
        #for i, z in enumerate(defect_slices):
        #    z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = get_skip_edges_from_nifty(z,i)

        #    delete_edges.extend(delete_edges_z)
        #    ignore_edges.extend(ignore_edges_z)

        #    assert len(skip_edges_z) == len(skip_ranges_z)
        #    skip_edges.extend(skip_edges_z)
        #    skip_ranges.extend(skip_ranges_z)
        #    skip_starts.extend(len(skip_edges_z) * [z-1])

        # parallel
        with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
            tasks = []
            for i,z in enumerate(defect_slices):
                tasks.append( executor.submit(get_skip_edges_from_nifty, z, i) )

            for task in tasks:

                z, delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = task.result()

                delete_edges.extend(delete_edges_z)
                ignore_edges.extend(ignore_edges_z)

                assert len(skip_edges_z) == len(skip_ranges_z)
                skip_edges.extend(skip_edges_z)
                skip_ranges.extend(skip_ranges_z)
                skip_starts.extend(len(skip_edges_z) * [z-1])

        delete_edges = np.unique(delete_edges).astype(np.uint32)
        uv_ids = np.delete(uv_ids,delete_edges,axis=0)
        workflow_logger.info("ModifiedAdjacency: deleted %i z-edges due to defects" % len(delete_edges) )

        skip_edges = np.array(skip_edges, dtype = np.uint32)
        skips_before_unique = skip_edges.shape[0]

        print skip_edges.shape
        # make the skip edges unique, keeping rows (see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array):
        skips_view = np.ascontiguousarray(skip_edges).view(np.dtype((np.void, skip_edges.dtype.itemsize * skip_edges.shape[1])))
        _, idx = np.unique(skips_view, return_index=True)
        skip_edges = skip_edges[idx]
        workflow_logger.info("ModifiedAdjacency: Removed %i duplicate skip edges" % (skips_before_unique - skip_edges.shape[0]) )

        skip_ranges = np.array(skip_ranges, dtype = np.uint32)
        skip_ranges = skip_ranges[idx]
        skip_starts = np.array(skip_starts, dtype = np.uint32)
        skip_starts = skip_starts[idx]
        assert skip_edges.shape[0] == skip_ranges.shape[0]
        assert skip_starts.shape[0] == skip_ranges.shape[0]
        workflow_logger.info("ModifiedAdjacency: introduced %i skip edges due to defects" % len(skip_edges) )

        # reorder the skip edges s.t. skip_starts are monotonically increasing
        sort_indices = np.argsort(skip_starts)
        skip_edges = skip_edges[sort_indices]
        skip_ranges = skip_ranges[sort_indices]
        skip_starts = skip_starts[sort_indices]
        # make sure that z is monotonically increasing (not strictly!)
        assert np.all(np.diff(skip_starts.astype(int)) >= 0), "Start index of skip edges must increase monotonically."

        ignore_edges = np.unique(ignore_edges).astype(np.uint32)
        workflow_logger.info("ModifiedAdjacency: found %i ignore edges due to defects" % len(ignore_edges) )

        # create the modified adjacency
        modified_adjacency = nifty.graph.UndirectedGraph(int(n_nodes))
        # insert original remaining edges
        modified_adjacency.insertEdges(uv_ids)
        # insert skip edges
        if skip_edges.size:
            modified_adjacency.insertEdges(skip_edges)
        n_edges_modified = modified_adjacency.numberOfEdges
        assert n_edges_modified == rag.numberOfEdges - delete_edges.shape[0] + skip_edges.shape[0], "%i, %i" % (n_edges_modified, rag.numberOfEdges - delete_edges.shape[0] + skip_edges.shape[0])
        workflow_logger.info("ModifiedAdjacency: Total number of edges in modified adjacency: %i" % n_edges_modified)

        out = self.output()
        out.write(modified_adjacency.serialize(), "modified_adjacency")
        out.write(skip_edges, "skip_edges")
        out.write(skip_starts, "skip_starts")
        out.write(skip_ranges, "skip_ranges")
        out.write(delete_edges, "delete_edges")
        out.write(ignore_edges, "ignore_edges")
        out.write(n_edges_modified, "n_edges_modified")

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "ModifiedAdjacency_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


# calculate and insert region features for the skip_edges
# and delete the delete_edges
class ModifiedRegionFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    def requires(self):
        return {'region_feats' : RegionFeatures(self.pathToInput, self.pathToSeg),
                'node_feats'   : RegionNodeFeatures(self.pathToInput, self.pathToSeg),
                'modified_adjacency' : ModifiedAdjacency(self.pathToSeg),
                'rag' : StackedRegionAdjacencyGraph(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()

        # the unmodified features
        node_feats = inp['node_feats']
        node_feats.open()
        region_feats = inp['region_feats']
        region_feats.open()

        # the modified edge connectivity
        modified_adjacency = inp['modified_adjacency']
        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        delete_edges = modified_adjacency.read('delete_edges')
        n_edges_modified = modified_adjacency.read('n_edges_modified')

        transition_edge = rag.totalNumberOfInSliceEdges

        # modify the features only if we have skip edges
        if skip_edges.size:

            assert skip_ranges.shape[0] == skip_edges.shape[0]
            assert delete_edges.min() >= transition_edge

            n_modified = region_feats.shape[0] - delete_edges.shape[0] + skip_edges.shape[0]
            n_feats = region_feats.shape[1] + 1 # we add an additional feature to indicate the skip range
            assert n_modified == n_edges_modified, "%i, %i" % (n_modified, n_edges_modified)

            out_shape = [n_modified, n_feats]
            chunk_shape = [2500,n_feats]
            out = self.output()
            out.open(out_shape, chunk_shape)

            # first, copy the xy-features, adding a 0 column indicating the skip range
            #out.write([0,0],
            #    np.c_[region_feats.read([0,0],[transition_edge,region_feats.shape[1]]),np.zeros(transition_edge)])

            out.write([0,0], np.c_[region_feats.read([0,0],[transition_edge,region_feats.shape[1]]),np.zeros(transition_edge)])

            print "In-shape: ", region_feats.shape
            print "Out-shape: ", out.shape
            print "Del edges:", delete_edges.shape

            # next, copy the z-edges, leaving out delete edges
            # add a column of 1s to indicate the skip range
            prev_edge = transition_edge
            total_copied = transition_edge

            consecutive_deletes = np.split(delete_edges,
                    np.where(np.diff(delete_edges) != 1)[0] + 1)

            # keep edges
            keep_edge_intervals = []
            if prev_edge != consecutive_deletes[0][0]:
                keep_edge_intervals.append([prev_edge, consecutive_deletes[0][0]])

            keep_edge_intervals.extend(
                [[consecutive_deletes[i][-1]+1, consecutive_deletes[i+1][0]] for i in xrange(len(consecutive_deletes)-1)])

            if consecutive_deletes[-1][-1] != region_feats.shape[0] - 1:
                keep_edge_intervals.append([consecutive_deletes[-1][-1]+1,region_feats.shape[0]])

            #print keep_edge_intervals
            for keep_start, keep_stop in keep_edge_intervals:
                #print keep_start, keep_stop
                n_copy = keep_stop - keep_start
                print keep_start, keep_stop
                assert n_copy > 0, str(n_copy)
                assert keep_stop <= region_feats.shape[0], "%i, %i" % (keep_stop, region_feats.shape[0])
                temp = np.c_[region_feats.read([keep_start,0], [keep_stop,region_feats.shape[1]]), np.ones(n_copy)]
                print temp.shape
                out.write([total_copied,0], temp)
                total_copied += n_copy

            assert total_copied == region_feats.shape[0] - delete_edges.shape[0], "%i, %i" % (total_copied, region_feats.shape[0] - delete_edges.shape[0])

            # finally, calculate the region features for the skip edges
            # features based on region statistics
            n_stat_feats = 16 # magic_nu...
            region_stats = node_feats.read([0,0],[node_feats.shape[0],n_stat_feats])

            fU = region_stats[skip_edges[:,0],:]
            fV = region_stats[skip_edges[:,1],:]

            skip_stat_feats = np.concatenate([np.minimum(fU,fV),
                np.maximum(fU,fV),
                np.abs(fU - fV),
                fU + fV], axis = 1)
            out.write([total_copied,0], skip_stat_feats)

            # features based on region center differences
            region_centers = node_feats.read([0,n_stat_feats],[node_feats.shape[0],node_feats.shape[1]])
            sU = region_centers[skip_edges[:,0],:]
            sV = region_centers[skip_edges[:,1],:]
            skip_center_feats = np.c_[(sU - sV)**2, skip_ranges]

            assert skip_center_feats.shape[0] == skip_stat_feats.shape[0]
            assert skip_center_feats.shape[1] == out.shape[1] - skip_stat_feats.shape[1]
            out.write([total_copied,skip_stat_feats.shape[1]], skip_center_feats)

        # we don't have any skip edges, so we only add an extra column, corresponding to the skip range (0 / 1)
        else:
            assert not skip_ranges.size
            assert not delete_edges.size

            n_edges = region_feats.shape[0]
            n_feats = region_feats.shape[1] + 1 # we add an additional feature to indicate the skip range
            out_shape = [n_edges, n_feats]
            chunk_shape = [2500,n_feats]
            out = self.output()
            out.open(out_shape, chunk_shape)

            # copy the xy-features, adding a 0 column indicating the skip range
            out.write([0,0],
                np.c_[region_feats.read([0,0],[transition_edge,region_feats.shape[1]]),np.zeros(transition_edge)])
            # copy the z-features, adding a 1 column indicating the skip range
            out.write([0,0],
                np.c_[region_feats.read([transition_edge,0],[n_edges,region_feats.shape[1]]),np.zeros(n_edges - transition_edge)])

        out.close()

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "ModifiedRegionFeatures_%s.h5" % (segFile,) )
        return HDF5VolumeTarget( save_path, 'float32' )


# calculate and insert edge features for the skip edges
# and delete the delete_edges
class ModifiedEdgeFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg = luigi.Parameter()

    keepOnlyXY = luigi.BoolParameter(default = False)
    keepOnlyZ = luigi.BoolParameter(default = False)

    def requires(self):
        return {'edge_feats' : EdgeFeatures(self.pathToInput, self.pathToSeg, self.keepOnlyXY, self.keepOnlyZ),
                'modified_adjacency' : ModifiedAdjacency(self.pathToSeg),
                'rag' : StackedRegionAdjacencyGraph(self.pathToSeg),
                'data' : InputData(self.pathToInput) }

    @run_decorator
    def run(self):

        inp = self.input()
        rag = inp['rag'].read()

        assert not (self.keepOnlyXY and self.keepOnlyZ)

        # the unmodified features
        edge_feats = inp['edge_feats']
        edge_feats.open()

        # the modified edge connectivity
        modified_adjacency = inp['modified_adjacency']
        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        delete_edges = modified_adjacency.read('delete_edges')
        n_edges_modified = modified_adjacency.read('n_edges_modified')

        # modify the features only if we have skip edges
        if skip_edges.size:
            assert skip_ranges.shape[0] == skip_edges.shape[0]

            transition_edge = rag.totalNumberOfInSliceEdges
            assert delete_edges.min() >= transition_edge

            n_feats = edge_feats.shape[1]
            if self.keepOnlyXY:
                n_modified = edge_feats.shape[0]
            else:
                n_modified = edge_feats.shape[0] - delete_edges.shape[0] + skip_edges.shape[0]

            if (not self.keepOnlyZ) and (not self.keepOnlyXY):
                assert n_modified == n_edges_modified, "%i, %i" % (n_modified, n_edges_modified)

            out_shape = [n_modified, n_feats]
            chunk_shape = [2500,n_feats]
            out = self.output()
            out.open(out_shape, chunk_shape)

            print "Out-shape:", out.shape

            # we copy the edge feats for xy-edges
            # don't do this if keepOnlyZ
            if not self.keepOnlyZ:
                out.write([0,0], edge_feats.read([0,0], [transition_edge,n_feats]))
                total_copied = transition_edge
                prev_edge    = transition_edge
            else:
                total_copied = 0
                prev_edge   = 0

            # copy the z-edges, leaving out delete edges and calculate features for the skip edges
            # don't do this if keepOnlyXY
            if not self.keepOnlyXY:

                if self.keepOnlyZ: # if only z features are present, we need to decrement the delete edges by the number of xy edges
                    delete_edges -= transition_edge
                    assert np.all(delete_edges > 0)

                consecutive_deletes = np.split(delete_edges,
                        np.where(np.diff(delete_edges) != 1)[0] + 1)

                # keep edges
                keep_edge_intervals = []
                if prev_edge != consecutive_deletes[0][0]:
                    keep_edge_intervals.append([prev_edge, consecutive_deletes[0][0]])

                keep_edge_intervals.extend(
                    [[consecutive_deletes[i][-1]+1, consecutive_deletes[i+1][0]] for i in xrange(len(consecutive_deletes)-1)])

                if consecutive_deletes[-1][-1] != edge_feats.shape[0] - 1:
                    keep_edge_intervals.append([consecutive_deletes[-1][-1]+1,edge_feats.shape[0]])

                print keep_edge_intervals
                for keep_start, keep_stop in keep_edge_intervals:
                    print keep_start, keep_stop
                    n_copy = keep_stop - keep_start
                    assert n_copy > 0, str(n_copy)
                    out.write([total_copied,0],
                        edge_feats.read([keep_start,0], [keep_stop,n_feats]) )
                    total_copied += n_copy

                if not total_copied == edge_feats.shape[0] - delete_edges.shape[0]:
                    ipdb.set_trace()

                assert total_copied == edge_feats.shape[0] - delete_edges.shape[0], "%i, %i" % (total_copied, edge_feats.shape[0] - delete_edges.shape[0])

                # compute skip features with nifty
                data = inp['data']
                data.open()

                skip_starts = modified_adjacency.read('skip_starts')
                #print np.unique(skip_starts)

                skip_feats = nifty.graph.rag.accumulateSkipEdgeFeaturesFromFilters(rag,
                        data.get(),
                        [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges], # skip_edges need to be passed as a list of pairs!
                        list(skip_ranges),
                        list(skip_starts),
                        PipelineParameter().nThreads )

                assert skip_feats.shape[0] == skip_edges.shape[0]
                assert skip_feats.shape[1] == n_feats
                out.write([total_copied,0], skip_feats)

        # we don't have any skip edges so we only copy
        else:
            assert not skip_ranges.size
            assert not delete_edges.size

            n_edges = edge_feats.shape[0]
            n_feats = edge_feats.shape[1]
            out_shape = [n_edges, n_feats]
            chunk_shape = [2500,n_feats]
            out = self.output()
            out.open(out_shape, chunk_shape)

            # copy all features
            out.write([0L,0L],edge_feats.read([0L,0L],edge_feats.shape))

        out.close()

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        inpFile = os.path.split(self.pathToInput)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "ModifiedEdgeFeatures_%s_%s" % (segFile,inpFile)  )
        if self.keepOnlyXY:
            save_path += '_xy'
        if self.keepOnlyZ:
            save_path += '_z'
        save_path += '.h5'
        return HDF5VolumeTarget( save_path, 'float32' )


class SkipEdgeLengths(luigi.Task):

    pathToSeg = luigi.Parameter()

    def requires(self):
        return {'rag' : StackedRegionAdjacencyGraph(self.pathToSeg),
                'modified_adjacency' : ModifiedAdjacency(self.pathToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()
        mod_adjacency = inp['modified_adjacency']
        skip_edges = mod_adjacency.read('skip_edges')
        skip_ranges = mod_adjacency.read('skip_ranges')
        skip_starts = mod_adjacency.read('skip_starts')

        skip_lens = nifty.graph.rag.getSkipEdgeLengths(rag,
                        [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges], # skip_edges need to be passed as a list of pairs!
                        list(skip_ranges),
                        list(skip_starts),
                        PipelineParameter().nThreads )
        skip_lens = np.array(skip_lens, dtype = 'uint32')
        assert skip_lens.shape[0] == skip_edges.shape[0]
        workflow_logger.info("SkipEdgeLengths: computed lens in range %i to %i" % (skip_lens.min(),skip_lens.max()))

        self.output().write(skip_lens, 'data')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "SkipEdgeLengths_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)
