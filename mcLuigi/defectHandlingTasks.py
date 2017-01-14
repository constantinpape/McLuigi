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
                'defects' : DefectSliceDetection(self.pathToSeg)} # need to tweek binThreshold w/ the default param

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
        save_path = os.path.join( PipelineParameter().cache, "DefectsToNodes_%s.h5" % (segFile,) )
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
        assert np.all(np.diff(nodes_z) >= 0), "Defected slice index is not increasing monotonically!"

        edge_offset = rag.totalNumberOfInSliceEdges
        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        # loop over the defect nodes and find the skip edges,
        # compute the modify adjacency

        skip_edges   = [] # the skip edges that run over the defects in z
        skip_ranges  = [] # z-distance of the skip edges
        skip_starts  = [] # starting slices of the skip edges
        delete_edges = [] # the z-edges between defected and non-defected nodes that are deleted from the graph
        ignore_edges = [] # the xy-edges between defected and non-defected nodes, that will be set to maximally repulsive weights

        prev_z = -1

        # get the original rag adjacency
        uv_ids = rag.uvIds()
        n_nodes = uv_ids.max() + 1

        # returns skip edges from nodes in lower slice to nodes in the above slices for the defected nodes
        # if nodes in above slice is also defected, recursively calls itself and goes one slice up
        def get_skip_edges(z_up, z_dn, bb_begin, bb_end, mask, nodes_dn, seg_dn):

            seg_up = seg.read([z_up] + bb_begin, [z_up+1] + bb_end).squeeze()
            nodes_up = vigra.analysis.unique(seg_up[mask])

            skip_range = z_up - z_dn
            new_skip_edges = []
            new_skip_ranges = []

            for u_dn in nodes_dn:
                # find coordinates of node in lower slice and interesect with mask
                where_dn = seg_dn == u_dn
                mask_dn = np.logical_and(where_dn, mask)
                # find intersecting nodes in upper slice and find the nodes with overlap
                connected_nodes = np.unique(seg_up[mask_dn])
                # if any of the nodes is defected go to the next slice
                # TODO Don't know if this is the best way to do it, but already much better than skipping the whole slice
                if np.any([nn in defect_nodes for nn in connected_nodes]):
                    new_skip_edges, new_skip_ranges = get_skip_edges(z_up+1,z_dn,bb_begin,bb_end,mask,nodes_dn,seg_dn)
                    break
                else:
                    new_skip_edges.extend( [[min(u_dn,cc),max(u_dn,cc)] for cc in connected_nodes] )
                    new_skip_ranges.extend( [skip_range for _ in xrange(len(connected_nodes))] )

            return new_skip_edges, new_skip_ranges

        for i, u in enumerate(defect_nodes):
            print i, '/', len(defect_nodes)
            z = long(nodes_z[i])

            # remove edges and find edges connecting defected to non defected nodes
            for v, edge_id in rag.nodeAdjacency(long(u)):
                # check if this is a z-edge and if it is, remove it from the adjacency
                if edge_id > edge_offset:
                    delete_edges.append(edge_id)
                elif v not in defect_nnodes: # otherwise, add it to the ignore edges, if it v is non-defected
                    ignore_edges.append(edge_id)

            # don't need skip edges for first (good) or last slice
            if z == 0 or z == seg.shape[0] - 1:
                continue

            if prev_z != z: # only read the segmentations if we haven't loaded it yet
                seg_z = seg.read([z,0L,0L],[z+1,ny,nx]).squeeze()

            # find the node coordinates and the corresponding bounding box and the mask projected to the bounding box
            where_node = np.where(seg_z==u)
            begin_u = [min(where_node[0]),min(where_node[1])]
            end_u   = [max(where_node[0])+1,max(where_node[1])+1]

            where_in_bb = (np.array(map(lambda x : x - begin_u[0], where_node[0]), dtype = np.uint32),
                    np.array(map(lambda x : x - begin_u[1], where_node[1]), dtype = np.uint32) )
            mask = np.zeros( tuple(map(lambda x,y: x - y, end_u, begin_u)), dtype = bool)
            mask[where_in_bb] = 1

            # find the lower nodes for skip connections
            seg_dn = seg.read([z-1] + begin_u, [z] + end_u).squeeze()
            nodes_dn = vigra.analysis.unique(seg_dn[mask])

            # we discard defected nodes in the lower slice (if present), because these were already taken care of
            # in previous iterations
            nodes_dn = np.array([nn for nn in nodes_dn if nn not in defect_nodes])

            if nodes_dn.size: # only do stuff if the array is not empty
                skip_edges_u, skip_ranges_u = get_skip_edges(z+1, z-1, begin_u, end_u, mask, nodes_dn, seg_dn)
                skip_edges.extend(skip_edges_u)
                skip_ranges.extend(skip_ranges_u)
                skip_starts.extend(len(skip_edges_u) * [z-1])

            prev_z = z

        assert len(skip_edges) == len(skip_ranges)

        delete_edges.sort()
        uv_ids = np.delete(uv_ids,delete_edges,axis=0)
        workflow_logger.info("ModifiedAdjacency: deleted %i z-edges due to defects" % (len(delete_edges),) )

        skip_edges = np.array(skip_edges, dtype = np.uint32)
        skip_ranges = np.array(skip_ranges, dtype = np.uint32)
        workflow_logger.info("ModifiedAdjacency: introduced %i skip edges due to defects" % (len(skip_edges),) )

        ignore_edges.sort()
        workflow_logger.info("ModifiedAdjacency: found %i ignore edges due to defects" % (len(ignore_edges),) )

        # create the modified adjacency
        modified_adjacency = nifty.graph.UndirectedGraph(int(n_nodes))
        # insert original remaining edges
        modified_adjacency.insertEdges(uv_ids)
        # insert skip edges
        modified_adjacency.insertEdges(skip_edges)
        workflow_logger.info("ModifiedAdjacency: Total number of edges in modified adjacency: %i" % modified_adjacency.numberOfEdges)

        out = self.output()
        out.write(modified_adjacency.serialize(), "modified_adjacency")
        out.write(skip_edges, "skip_edges")
        out.write(skip_starts, "skip_starts")
        out.write(skip_ranges, "skip_ranges")
        out.write(delete_edges, "delete_edges")
        out.write(ignore_edges, "ignore_edges")

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

        transition_edge = rag.totalNumberOfInSliceEdges
        assert delete_edges.min() >= transition_edge

        n_modified = region_feats.shape[0] - delete_edges.shape[0] + skip_edges.shape[0]
        n_feats = region_feats.shape[1] + 1 # we add an additional feature to indicate the skip range
        out_shape = [n_modified, n_feats]
        chunk_shape = [2500,n_feats]
        out = self.output()
        out.open(out_shape, chunk_shape)

        # first, copy the xy-features, adding a 0 column indicating the skip range
        out.write([0,0],
            np.c_[region_feats.read([0,0],[transition_edge,region_feats.shape[1]]),np.zeros(transition_edge)])

        # next, copy the z-edges, leaving out delete edges
        # add a column of 1s to indicate the skip range
        prev_edge = transition_edge
        total_copied = transition_edge
        for del_edge in delete_edges:
            if del_edge == prev_edge:
                prev_edge = del_edge + 1
                continue
            n_copy = del_edge - prev_edge
            out.write([total_copied,0],
                np.c_[region_feats.read([prev_edge,0],[del_edge,region_feats.shape[1]]),np.ones(n_copy)] )
            prev_edge = del_edge + 1 # we skip over the del_edge
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

        transition_edge = rag.totalNumberOfInSliceEdges
        assert delete_edges.min() >= transition_edge

        n_feats = edge_feats.shape[1]
        if self.keepOnlyXY:
            n_modified = edge_feats.shape[0]
        else:
            n_modified = edge_feats.shape[0] - delete_edges.shape[0] + skip_edges.shape[0]

        out_shape = [n_modified, n_feats]
        chunk_shape = [2500,n_feats]
        out = self.output()
        out.open(out_shape, chunk_shape)

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

            for del_edge in delete_edges:
                if del_edge == prev_edge:
                    prev_edge = del_edge + 1
                    continue
                n_copy = del_edge - prev_edge
                #print [total_copied,0]
                #print [prev_edge,0]
                #print [del_edge,n_feats]
                out.write([total_copied,0],
                    edge_feats.read([prev_edge,0], [del_edge,n_feats]) )
                prev_edge = del_edge + 1 # we skip over the del_edge
                total_copied += n_copy

            assert total_copied == edge_feats.shape[0] - delete_edges.shape[0], "%i, %i" % (total_copied, edge_feats.shape[0] - delete_edges.shape[0])

            # compute skip features with nifty
            data = inp['data']
            data.open()

            skip_starts = modified_adjacency.read('skip_starts')
            #print np.unique(skip_starts)

            skip_feats = nifty.graph.rag.accumulateSkipEdgeFeaturesFromFilters(rag,
                    data.get(),
                    [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges], # skip_edges need to be passed as a list of pairs!
                    skip_ranges,
                    skip_starts,
                    PipelineParameter().nThreads )
            assert skip_feats.shape[0] == skip_edges.shape[0]
            assert skip_feats.shape[1] == n_feats
            out.write([total_copied,0], skip_feats)

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


# TODO implement
# replace the segmentation in completely defected slices with adjacent slice
class PostprocessDefectedSlices(luigi.Task):

    def requires(self):
        pass
