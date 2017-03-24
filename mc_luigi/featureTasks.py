# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from customTargets import HDF5DataTarget, HDF5VolumeTarget
from dataTasks import InputData, StackedRegionAdjacencyGraph, ExternalSegmentation
from defectHandlingTasks import ModifiedAdjacency

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import json

import os
import time
import numpy as np
import vigra

from concurrent import futures

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



class RegionNodeFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    def requires(self):
        return {
                "data" : InputData(self.pathToInput),
                "seg" : ExternalSegmentation(self.pathToSeg),
                "rag" : StackedRegionAdjacencyGraph(self.pathToSeg)
                }

    @run_decorator
    def run(self):

        inp = self.input()

        data = inp["data"]
        seg  = inp["seg"]

        data.open()
        seg.open()

        shape = data.shape()

        assert shape == seg.shape(), str(shape) + " , " + str(seg.shape())

        minMaxNodeSlice = inp['rag'].readKey('minMaxLabelPerSlice').astype('uint32')
        nNodes = inp['rag'].readKey('numberOfNodes')
        nFeats = 20

        # list of the region statistics, that we want to extract
        # drop te Histogram, because it blows up the feature space...
        statistics =  [ "Count", "Kurtosis", #Histogram
                        "Maximum", "Minimum", "Quantiles",
                        "RegionRadii", "Skewness", "Sum",
                        "Variance", "Weighted<RegionCenter>", "RegionCenter"]

        out = self.output()
        out_shape = [nNodes, nFeats]
        chunk_shape = [min(5000,nNodes), out_shape[1]]
        out.open(out_shape, chunk_shape)

        # get region statistics with the vigra region feature extractor for a single slice
        def extractRegionStatsSlice(start, end, z):

            minNode, maxNode = minMaxNodeSlice[z,0], minMaxNodeSlice[z,1]

            dataSlice = data.read(start,end).squeeze().astype('float32')
            segSlice  = seg.read(start,end).squeeze() - minNode

            extractor = vigra.analysis.extractRegionFeatures(dataSlice, segSlice, features = statistics )

            regionStatisticsSlice = []

            for statName in statistics:
                stat = extractor[statName]
                if stat.ndim == 1:
                    regionStatisticsSlice.append(stat[:,None])
                else:
                    regionStatisticsSlice.append(stat)

            regionStatisticsSlice = np.nan_to_num(
                    np.concatenate(regionStatisticsSlice, axis = 1).astype('float32') )

            startOut = [ long(minNode) ,0L]
            assert regionStatisticsSlice.shape[0] == maxNode + 1 - minNode
            out.write(startOut, regionStatisticsSlice)
            return True

        # sequential for debugging
        #results = []
        #for z in xrange(shape[0]):
        #    start = [z,0,0]
        #    end   = [z+1,shape[1],shape[2]]
        #    results.append( extractRegionStatsSlice( start, end, z) )

        # parallel
        nWorkers = min( shape[0], PipelineParameter().nThreads )
        #nWorkers = 1
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for z in xrange(shape[0]):
                start = [z,0,0]
                end   = [z+1,shape[1],shape[2]]
                tasks.append( executor.submit(extractRegionStatsSlice, start, end, z) )
        results = [task.result() for task in tasks]

        out.close()

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "RegionNodeFeatures_%s.h5" % (segFile,) )
        return HDF5VolumeTarget( save_path, 'float32' )


class RegionFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    # TODO have to rethink this if we include lifted multicut
    def requires(self):
        required_tasks = {"rag" : StackedRegionAdjacencyGraph(self.pathToSeg),
                "node_feats" : RegionNodeFeatures(self.pathToInput,self.pathToSeg)}
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):

        inp = self.input()
        out = self.output()
        nodeFeatsFile = inp["node_feats"]
        nodeFeatsFile.open()
        nodeFeats = nodeFeatsFile.read([0,0], nodeFeatsFile.shape())

        if PipelineParameter().defectPipeline:
            modified_adjacency = inp['modified_adjacency']
            if modified_adjacency.read('has_defects'):
                self._compute_modified_feats(nodeFeats, inp, out)
            else:
                self._compute_standard_feats(nodeFeats, inp, out)
        else:
            self._compute_standard_feats(nodeFeats, inp, out)

        nodeFeatsFile.close()
        out.close()


    def _compute_feats_from_uvs(self,
            nodeFeats,
            uvIds,
            key,
            out,
            skipRanges = None):

        if not isinstance(skipRanges, np.ndarray):
            assert skipRanges == None

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs called with key: %s" % key)
        nEdges = uvIds.shape[0]
        # magic 16 = number of regionStatistics that are combined by min, max, sum and absdiff
        nStatFeats = 16

        nFeats = 4*nStatFeats + 4
        if isinstance(skipRanges, np.ndarray):
            nFeats += 1
        out_shape = [nEdges, nFeats]
        chunk_shape = [2500, out_shape[1]]
        out.open(out_shape, chunk_shape, key)

        # the statistic features that are combined by min, max, sum and absdiff
        stats = nodeFeats[:,:nStatFeats]
        # the center features that are combined by quadratic euclidean distance
        centers = nodeFeats[:,nStatFeats:]

        def quadratic_euclidean_dist(x,y):
            return np.square(np.subtract(x, y))
        def absdiff(x,y):
            return np.abs( np.subtract(x,y) )
        combine = (np.minimum, np.maximum, absdiff, np.add)

        def feats_for_subset(uvsSub, edge_offset):
            fU = stats[uvsSub[:,0],:]
            fV = stats[uvsSub[:,1],:]
            feats_sub = [comb(fU,fV) for comb in combine]
            sU = centers[uvsSub[:,0],:]
            sV = centers[uvsSub[:,1],:]
            feats_sub.append(quadratic_euclidean_dist(sU, sV))
            feats_sub = np.concatenate(feats_sub, axis = 1)
            out.write( [edge_offset, 0], feats_sub, key)
            #print "Start edge: ", edge_offset, "/", nEdges, "done"
            return True

        # TODO maybe some tweeking can speed this up further
        # we should tune nSplits s.t. edgeStart - edgeStop is a multiple of chunks!
        # maybe less threads could also help ?!
        nWorkers = PipelineParameter().nThreads
        #nWorkers = 10
        # we split the edges in 500 blocks
        nSplits = 500
        with futures.ThreadPoolExecutor(max_workers = nWorkers) as executor:
            tasks = []
            for ii in xrange(nSplits):
                edgeStart = int(float(ii) / nSplits * nEdges)
                edgeStop  = nEdges if ii == nSplits-1 else int(float(ii+1) / nSplits  * nEdges)
                tasks.append( executor.submit(feats_for_subset,
                    uvIds[edgeStart:edgeStop,:],
                    edgeStart ) )
            res = [t.result() for t in tasks]

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs done.")

        if isinstance(skipRanges, np.ndarray):
            assert skipRanges.shape == (nEdges,)
            out.write([0,nFeats-1], skipRanges[:,None], key)


    def _compute_standard_feats(self, nodeFeats, inp, out):

        rag = inp['rag']
        uvIds = rag.readKey('uvIds')
        transitionEdge = rag.readKey('totalNumberOfInSliceEdges')

        # xy-feature
        self._compute_feats_from_uvs(nodeFeats, uvIds[:transitionEdge], "features_xy", out)
        # z-feature
        self._compute_feats_from_uvs(nodeFeats, uvIds[transitionEdge:], "features_z", out)


    # calculate and insert region features for the skip_edges
    # and delete the delete_edges
    def _compute_modified_feats(self, nodeFeats, inp, out):

        rag = inp['rag']
        modified_adjacency = inp['modified_adjacency']
        uvIds = rag.readKey('uvIds')
        transition_edge = rag.readKey('totalNumberOfInSliceEdges')

        # compute the standard xy-features with additional ranges
        self._compute_feats_from_uvs( nodeFeats,
                uvIds[:transition_edge],
                'features_xy',
                out,
                np.zeros(transition_edge, dtype = 'float32') )

        # compute the z-features with proper edges deleted from uv-ids
        delete_edges = modified_adjacency.read('delete_edges')
        uvsZ = uvIds[transition_edge:]
        if delete_edges.size:
            assert delete_edges.min() >= transition_edge
            delete_edges -= transition_edge
            uvsZ = np.delete(uvsZ, delete_edges, axis = 0)
        self._compute_feats_from_uvs( nodeFeats,
                uvsZ,
                'features_z',
                out,
                np.ones( uvsZ.shape[0], dtype = 'float32') )

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        assert skip_ranges.shape[0] == skip_edges.shape[0]

        # if we have skip edges, compute features for them
        if skip_edges.size:
            self._compute_feats_from_uvs(nodeFeats,
                    skip_edges,
                    'features_skip',
                    out,
                    skip_ranges)


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "RegionFeatures_" )
        if PipelineParameter().defectPipeline:
            save_path += "modified_%s.h5" % segFile
        else:
            save_path += "standard_%s.h5" % segFile
        return HDF5VolumeTarget( save_path, 'float32' )


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()
    keepOnlyXY = luigi.BoolParameter(default = False)
    keepOnlyZ = luigi.BoolParameter(default = False)

    # For now we can't set these any more, needs to be passed to C++ somehow
    #filterNames = luigi.ListParameter(default = [ "gaussianSmoothing", "hessianOfGaussianEigenvalues", "laplacianOfGaussian"] )
    #sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )

    def requires(self):
        required_tasks = {'rag' : StackedRegionAdjacencyGraph(self.pathToSeg),
                'data' : InputData(self.pathToInput)}
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):
        assert not(self.keepOnlyXY and self.keepOnlyZ)

        inp = self.input()
        rag = inp['rag'].read()
        dataFile = inp['data']
        dataFile.open()
        data = dataFile.get()
        out = self.output()

        if PipelineParameter().defectPipeline:
            modified_adjacency = inp['modified_adjacency']
            if modified_adjacency.read('has_defects'):
                self._compute_modified_feats(data, rag, modified_adjacency, out)
            else:
                self._compute_standard_feats(data, rag, out)
        else:
            self._compute_standard_feats(data, rag, out)

        out.close()
        dataFile.close()

        if PipelineParameter().defectPipeline:
            if modified_adjacency.read('has_defects'):
                self._postprocess_modified_output(out)


    def _compute_standard_feats(self, data, rag, out):

        workflow_logger.info("EdgeFeatures: _compute_standard_feats called.")
        nXYEdges = rag.totalNumberOfInSliceEdges if not self.keepOnlyZ else 0
        nZEdges  = rag.totalNumberOfInBetweenSliceEdges if not self.keepOnlyXY else 0

        # 9 * 12 = number of features per edge / would be nice not to hard code this here...
        nFeats = 9*12
        out_shape_xy    = [nXYEdges, nFeats]
        chunk_shape_xy  = [min(2500,nXYEdges), nFeats]
        out_shape_z    = [nZEdges, nFeats]
        chunk_shape_z  = [min(2500,nZEdges), nFeats]

        out.open(out_shape_xy, chunk_shape_xy, 'features_xy')
        out.open(out_shape_z, chunk_shape_z, 'features_z')

        nifty.graph.rag.accumulateEdgeFeaturesFromFilters(rag,
                data,
                out.get('features_xy'),
                out.get('features_z'),
                self.keepOnlyXY,
                self.keepOnlyZ,
                PipelineParameter().nThreads)
        workflow_logger.info("EdgeFeatures: _compute_standard_feats done.")


    def _compute_modified_feats(self, data, rag, modified_adjacency, out):

        workflow_logger.info("EdgeFeatures: _compute_modified_feats called.")
        # first, compute the standard feats
        self._compute_standard_feats(data, rag, out)

        transition_edge = rag.totalNumberOfInSliceEdges
        # copy the z-features we keep and delete the ones that are not needed
        delete_edges = modified_adjacency.read('delete_edges')
        if delete_edges.size and not self.keepOnlyXY:
            assert delete_edges.min() >= transition_edge
            # we substract the transition edge, because we count from the begin of z edges
            delete_edges -= transition_edge

            standard_feat_shape = out.shape('features_z')

            n_modified = standard_feat_shape[0] - delete_edges.shape[0]
            n_feats    = standard_feat_shape[1]

            out_shape   = [n_modified, n_feats]
            chunk_shape = [min(2500,n_modified),n_feats]
            out.open(out_shape, chunk_shape, 'features_z_keep')

            # find all edges with continuous indices that will be deleted
            consecutive_deletes = np.split(delete_edges,
                    np.where(np.diff(delete_edges) != 1)[0] + 1)

            prev_edge, total_copied = 0, 0
            # find the interval of edges to keep
            keep_edge_intervals = []
            if prev_edge != consecutive_deletes[0][0]:
                keep_edge_intervals.append([prev_edge, consecutive_deletes[0][0]])

            keep_edge_intervals.extend(
                [[consecutive_deletes[i][-1]+1, consecutive_deletes[i+1][0]] for i in xrange(len(consecutive_deletes)-1)])

            if consecutive_deletes[-1][-1] != standard_feat_shape[0] - 1:
                keep_edge_intervals.append([consecutive_deletes[-1][-1]+1,standard_feat_shape[0]])

            for keep_start, keep_stop in keep_edge_intervals:
                n_copy = keep_stop - keep_start
                assert n_copy > 0, str(n_copy)
                out.write([total_copied,0],
                    out.read([keep_start,0], [keep_stop,n_feats], 'features_z'),
                    'features_z_keep')
                total_copied += n_copy

            assert total_copied == standard_feat_shape[0] - delete_edges.shape[0], "%i, %i" % (total_copied, standard_feat_shape[0] - delete_edges.shape[0])

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        skip_starts = modified_adjacency.read('skip_starts')
        assert skip_ranges.shape[0] == skip_edges.shape[0]
        assert skip_starts.shape[0] == skip_edges.shape[0]

        # modify the features only if we have skip edges
        if skip_edges.size:

            # TODO i/o in nifty to speed up calculation
            skip_feats = nifty.graph.rag.accumulateSkipEdgeFeaturesFromFilters(rag,
                    data,
                    [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges], # skip_edges need to be passed as a list of pairs!
                    list(skip_ranges),
                    list(skip_starts),
                    PipelineParameter().nThreads )

            assert skip_feats.shape[0] == skip_edges.shape[0]
            assert skip_feats.shape[1] == n_feats
            print "Going to write skip edges" # could use h5py too
            out.open( skip_feats.shape,
                      [min(2500,skip_feats.shape[0]), skip_feats.shape[1]],
                      'features_skip')
            out.write([0,0], skip_feats, 'features_skip')


    # TODO check that this actually does what it is supposed to
    def _postprocess_modified_output(self, out):
        import h5py
        with h5py.File(out.path) as f:
            # first remove the standard features_z
            del f['features_z']
            # next, rename the features_z_keep to features_z
            # TODO make sure that this does not copy any data!
            f['features_z'] = f['features_z_keep']
            del f['features_z_keep']


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        inpFile = os.path.split(self.pathToInput)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "EdgeFeatures_%s_%s" % (segFile,inpFile)  )
        if PipelineParameter().defectPipeline:
            save_path += '_modified'
        else:
            save_path += '_standard'
        if self.keepOnlyXY:
            save_path += '_xy'
        if self.keepOnlyZ:
            save_path += '_z'
        save_path += '.h5'
        return HDF5VolumeTarget( save_path, 'float32' )


# TODO in nifty ??
# the edgeLens are implemented, rest will be more tricky and is not that helpful anyway...

#class TopologyFeatures(luigi.Task):
#
#    PathToSeg = luigi.Parameter()
#    Use2dFeatures = luigi.BoolParameter(default = True)
#
#    def requires(self):
#        if self.Use2dFeatures:
#            return {"Seg" : ExternalSegmentationLabeled(self.PathToSeg) , "RAG" : RegionAdjacencyGraph(self.PathToSeg),
#                    "EdgeIndications" : EdgeIndications(self.PathToSeg) }
#        else:
#            return {"Seg" : ExternalSegmentationLabeled(self.PathToSeg) , "RAG" : RegionAdjacencyGraph(self.PathToSeg)}
#
#    # Features from edge_topology
#    #def topology_features(self, seg_id, use_2d_edges):
#    def run(self):
#
#        t_feats = time.time()
#
#        rag = self.input()["RAG"].read()
#        seg = self.input()["Seg"].read()
#
#        if self.Use2dFeatures:
#            n_feats = 7
#        else:
#            n_feats = 1
#
#        n_edges = rag.edgeNum
#        topology_features = np.zeros( (n_edges, n_feats) )
#
#        # length / area of the edge
#        edge_lens = rag.edgeLengths()
#        assert edge_lens.shape[0] == n_edges
#        topology_features[:,0] = edge_lens
#
#        # extra feats for z-edges in 2,5 d
#        if self.Use2dFeatures:
#
#            # edge indications
#            edge_indications = self.input()["EdgeIndications"].read()
#            assert edge_indications.shape[0] == n_edges
#            topology_features[:,1] = edge_indications
#
#            # region sizes to build some features
#            statistics =  [ "Count", "RegionCenter" ]
#
#            extractor = vigra.analysis.extractRegionFeatures(
#                    np.zeros_like(seg, dtype = np.float32), # dummy input
#                    seg, features = statistics )
#
#            z_mask = edge_indications == 0
#
#            sizes = extractor["Count"]
#            uvIds = np.sort( rag.uvIds(), axis = 1)
#            sizes_u = sizes[ uvIds[:,0] ]
#            sizes_v = sizes[ uvIds[:,1] ]
#            # union = size_up + size_dn - intersect
#            unions  = sizes_u + sizes_v - edge_lens
#            # Union features
#            topology_features[:,2][z_mask] = unions[z_mask]
#            # IoU features
#            topology_features[:,3][z_mask] = edge_lens[z_mask] / unions[z_mask]
#
#            # segment shape features
#            seg_coordinates = extractor["RegionCenter"]
#            len_bounds      = np.zeros(rag.nodeNum)
#            # TODO no loop ?! or CPP
#            # iterate over the nodes, to get the boundary length of each node
#            for n in rag.nodeIter():
#                node_z = seg_coordinates[n.id][2]
#                for arc in rag.incEdgeIter(n):
#                    edge = rag.edgeFromArc(arc)
#                    edge_c = rag.edgeCoordinates(edge)
#                    # only edges in the same slice!
#                    if edge_c[0,2] == node_z:
#                        len_bounds[n.id] += edge_lens[edge.id]
#            # shape feature = Area / Circumference
#            shape_feats_u = sizes_u / len_bounds[uvIds[:,0]]
#            shape_feats_v = sizes_v / len_bounds[uvIds[:,1]]
#            # combine w/ min, max, absdiff
#            print shape_feats_u[z_mask].shape
#            print shape_feats_v[z_mask].shape
#            topology_features[:,4][z_mask] = np.minimum(
#                    shape_feats_u[z_mask], shape_feats_v[z_mask])
#            topology_features[:,5][z_mask] = np.maximum(
#                    shape_feats_u[z_mask], shape_feats_v[z_mask])
#            topology_features[:,6][z_mask] = np.absolute(
#                    shape_feats_u[z_mask] - shape_feats_v[z_mask])
#
#        t_feats = time.time() - t_feats
#        workflow_logger.info("Calculated Topology Features in: " + str(t_feats) + " s")
#
#        self.output().write(topology_features)
#
#    def output(self):
#        return HDF5Target( os.path.join( PipelineParameter().cache, "TopologyFeatures.h5" ) )
