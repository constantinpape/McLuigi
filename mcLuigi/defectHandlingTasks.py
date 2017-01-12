# Multicut Pipeline implemented with luigi
# Taksks for defect and handling
import luigi

from customTargets import HDF5DataTarget, HDF5VolumeTarget
from dataTasks import InputData, ExternalSegmentation, StackedRegionAdjacencyGraph
from defectDetectionTasks import DefectSliceDetection

from pipelineParameter import PipelineParameter
from tools import config_logger

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

    def run(self):
        workflow_logger.info("Computing DefectToNodes")

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

        workflow_logger.info("Found %i defected nodes" % (len(defect_nodes)))

        self.output().write(np.array(defect_nodes,dtype='uint32'),'defect_nodes')
        self.output().write(np.array(nodes_z,dtype='uint32'),'nodes_z')

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "DefectsToNodes_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


# TODO handle consecutive defected slices for individual patch predictions
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
    def run(self):
        inp = self.input()
        rag = inp['rag'].read()
        defect_nodes = inp['defect_nodes'].read('defect_nodes')
        nodes_z = inp['defect_nodes'].read('nodes_z')
        seg = inp['seg']
        seg.open()

        workflow_logger.info("Computing ModiifiedAdjacency")

        edge_offset = rag.totalNumberOfInSliceEdges
        ny = long(seg.shape[1])
        nx = long(seg.shape[2])

        # loop over the defect nodes and find the skip edges,
        # compute the modify adjacency
        skip_edges = []
        skip_ranges = []

        # get the rag adjacency
        uv_ids = rag.uvIds()
        n_nodes = uv_ids.max() + 1

        # TODO this is fine for now, since we mark whole slices as defected anyway,
        # but we have individual defect patches need to come up with something more clever!
        # if we have defects in multiple consecutive slices we only consider the lowest one and skip
        # over the following defected slices
        slices_with_defect = vigra.analysis.unique(nodes_z)
        consecutive_defects = np.split(slices_with_defect, np.where(np.diff(slices_with_defect) != 1)[0]+1)
        lowest_defect_slices = [consec[0] for consec in consecutive_defects if len(consec) > 1]
        skip_slices          = [cc for c in consec for consec in consecutive_defects if len(consec) > 1]
        next_good_slices     = [consec[-1] + 1 for consec in consecutive_defects if len(consec) > 1]
        assert len(lowest_defect_slices) == len(next_good_slices)
        lowest_to_good       = {lowest_defect_slices[ii] : next_good_slices[ii] for ii in xrange(len(next_good_slices))}

        # fix the corner case that we have a succession of bad slices at the bottom
        lowest_good_slice = 0
        if 0 in lowest_to_good:
            lowest_good_slice = lowest_to_good[0]

        workflow_logger.info("Found %i consecutive defected slices" % (len(lowest_defect_slices)))
        workflow_logger.info("Lowest slices to next good slices: " + str(lowest_to_good))
        workflow_logger.info("Skip slices: " + str(skip_slices))

        # edges to be deleted
        delete_edges = []
        prev_z = -1

        # TODO can we speed this up somehow? -> parallelize or move impl to c++
        print "Processing defected nodes:"
        for i, u in enumerate(defect_nodes):
            print i, '/', len(defect_nodes)
            z = long(nodes_z[i])

            # remove edges
            for v, edge_id in rag.nodeAdjacency(long(u)):
                # check if this is a z-edge and if it is, remove it from the adjacency
                if edge_id > edge_offset:
                    delete_edges.append(edge_id)

            if z in skip_slices: #skip over consecutive defected slices that are not lowest
                continue
            if z in lowest_defect_slices: # if we are at lowest slice of consecutive defected slices, need to skip further
                z_up = lowest_to_good[z]
                if z_up >= seg.shape[0]: # next good slice is out of range
                    continue
            else:
                z_up = long(z + 1)
            z_dn = long(z - 1)

            # don't need skip edges for first (good) or last slice
            if z == lowest_good_slice or z == seg.shape[0] - 1:
                continue

            # find skip edges
            skip_range = z_up - z_dn

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

            # find the upper and lower nodes for skip connections
            seg_dn = seg.read([z_dn] + begin_u, [z] + end_u).squeeze()
            seg_up = seg.read([z_up] + begin_u, [z_up+1] + end_u).squeeze()

            # connect up and dn nodes that have overlap in the mask
            nodes_dn = vigra.analysis.unique(seg_dn[mask])
            nodes_up = vigra.analysis.unique(seg_up[mask])

            # TODO need to handle the possibility that the up nodes could be defected later on here
            # that shouldn't be too hard, because we have already processed every thing in the lower part
            for u_dn in nodes_dn:
                # find coordinates of node in lower slice and interesect with mask
                where_dn = seg_dn == u_dn
                mask_dn = np.logical_and(where_dn, mask)
                # find intersecting nodes in upper slice and add skip edges
                connected_nodes = np.unique(seg_up[mask_dn])
                skip_edges.extend( [[min(u_dn,cc),max(u_dn,cc)] for cc in connected_nodes] )
                skip_ranges.extend( [skip_range for _ in xrange(len(connected_nodes)) ] )

            prev_z = z

        assert len(skip_edges) == len(skip_ranges)

        delete_edges.sort()
        uv_ids = np.delete(uv_ids,delete_edges,axis=0)
        workflow_logger.info("Deleted %i z-edges due to defects" % (len(delete_edges),) )

        skip_edges = np.array(skip_edges, dtype = np.uint32)
        skip_ranges = np.array(skip_ranges, dtype = np.uint32)
        workflow_logger.info("Introduced %i skip edges due to defects" % (len(skip_edges),) )

        # create the modified adjacency
        modified_adjacency = nifty.graph.UndirectedGraph(int(n_nodes))
        # insert original remaining edges
        modified_adjacency.insertEdges(uv_ids)
        # insert skip edges
        modified_adjacency.insertEdges(skip_edges)

        out = self.output()
        out.write(modified_adjacency.serialize(), "modified_adjacency")
        out.write(skip_edges, "skip_edges")
        out.write(skip_ranges, "skip_ranges")

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "ModifiedAdjacency_%s.h5" % (segFile,) )
        return HDF5DataTarget(save_path)


# TODO implement
# calculate and insert features for the skip edges
class ModifiedFeatures(luigi.Task):

    def requires(self):
        pass


# TODO implement
# modify the mc problem by changing the graph to the 'ModifiedAdjacency'
# and setting xy-edges that connect defected with non-defected nodes to be maximal repulsive
class ModifiedMcProblem(luigi.Task):

    def requires(self):
        pass


# TODO implement
# replace the segmentation in completely defected slices with adjacent slice
class PostprocessDefectedSlices(luigi.Task):

    def requires(self):
        pass
