# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from customTargets import HDF5VolumeTarget
from dataTasks import InputData, StackedRegionAdjacencyGraph, ExternalSegmentation
from defectHandlingTasks import ModifiedAdjacency

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator

import logging
import os
import numpy as np
import vigra

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


class RegionNodeFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    def requires(self):
        return {
            "data": InputData(self.pathToInput),
            "seg": ExternalSegmentation(self.pathToSeg),
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg)
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

        min_max_node = inp['rag'].readKey('minMaxLabelPerSlice').astype('uint32')
        n_nodes = inp['rag'].readKey('numberOfNodes')
        n_feats = 20

        # list of the region statistics, that we want to extract
        # drop te Histogram, because it blows up the feature space...
        statistics = [
            "Count", "Kurtosis",  # Histogram
            "Maximum", "Minimum", "Quantiles",
            "RegionRadii", "Skewness", "Sum",
            "Variance", "Weighted<RegionCenter>", "RegionCenter"
        ]

        out = self.output()
        out_shape = [n_nodes, n_feats]
        chunk_shape = [min(5000, n_nodes), out_shape[1]]
        out.open(out_shape, chunk_shape)

        # get region statistics with the vigra region feature extractor for a single slice
        def extract_stats_slice(start, end, z):

            min_node, max_node = min_max_node[z, 0], min_max_node[z, 1]

            data_slice = data.read(start, end).squeeze().astype('float32')
            seg_slice  = seg.read(start, end).squeeze() - min_node

            extractor = vigra.analysis.extractRegionFeatures(data_slice, seg_slice, features=statistics)

            region_stats_slice = []

            for stat_name in statistics:
                stat = extractor[stat_name]
                if stat.ndim == 1:
                    region_stats_slice.append(stat[:, None])
                else:
                    region_stats_slice.append(stat)

            region_stats_slice = np.nan_to_num(
                np.concatenate(region_stats_slice, axis=1).astype('float32')
            )

            assert region_stats_slice.shape[0] == max_node + 1 - min_node
            out.write([long(min_node), 0L], region_stats_slice)
            return True

        # parallel
        n_workers = min(shape[0], PipelineParameter().nThreads)
        # n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = []
            for z in xrange(shape[0]):
                start = [z, 0, 0]
                end   = [z + 1, shape[1], shape[2]]
                tasks.append(executor.submit(extract_stats_slice, start, end, z))
        [task.result() for task in tasks]

        out.close()

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "RegionNodeFeatures_%s.h5" % seg_file)
        return HDF5VolumeTarget(save_path, 'float32')


class RegionFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    # TODO have to rethink this if we include lifted multicut
    def requires(self):
        required_tasks = {
            "rag": StackedRegionAdjacencyGraph(self.pathToSeg),
            "node_feats": RegionNodeFeatures(self.pathToInput, self.pathToSeg)
        }
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):

        inp = self.input()
        out = self.output()
        node_feats_file = inp["node_feats"]
        node_feats_file.open()
        node_feats = node_feats_file.read([0, 0], node_feats_file.shape())

        if PipelineParameter().defectPipeline:
            modified_adjacency = inp['modified_adjacency']
            if modified_adjacency.read('has_defects'):
                self._compute_modified_feats(node_feats, inp, out)
            else:
                self._compute_standard_feats(node_feats, inp, out)
        else:
            self._compute_standard_feats(node_feats, inp, out)

        node_feats_file.close()
        out.close()

    def _compute_feats_from_uvs(
        self,
        node_feats,
        uv_ids,
        key,
        out,
        skip_ranges=None
    ):

        if not isinstance(skip_ranges, np.ndarray):
            assert skip_ranges is None

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs called with key: %s" % key)
        n_edges = uv_ids.shape[0]
        # magic 16 = number of regionStatistics that are combined by min, max, sum and absdiff
        nStatFeats = 16

        n_feats = 4 * nStatFeats + 4
        if isinstance(skip_ranges, np.ndarray):
            n_feats += 1
        out_shape = [n_edges, n_feats]
        chunk_shape = [2500, out_shape[1]]
        out.open(out_shape, chunk_shape, key)

        # the statistic features that are combined by min, max, sum and absdiff
        stats = node_feats[:, :nStatFeats]
        # the center features that are combined by quadratic euclidean distance
        centers = node_feats[:, nStatFeats:]

        def quadratic_euclidean_dist(x, y):
            return np.square(np.subtract(x, y))

        def absdiff(x, y):
            return np.abs(np.subtract(x, y))

        combine = (np.minimum, np.maximum, absdiff, np.add)

        def feats_for_subset(uvs_sub, edge_offset):
            fU = stats[uvs_sub[:, 0], :]
            fV = stats[uvs_sub[:, 1], :]
            feats_sub = [comb(fU, fV) for comb in combine]
            sU = centers[uvs_sub[:, 0], :]
            sV = centers[uvs_sub[:, 1], :]
            feats_sub.append(quadratic_euclidean_dist(sU, sV))
            feats_sub = np.concatenate(feats_sub, axis=1)
            out.write([edge_offset, 0], feats_sub, key)
            return True

        # TODO maybe some tweeking can speed this up further
        # we should tune nSplits s.t. edgeStart - edgeStop is a multiple of chunks!
        # maybe less threads could also help ?!
        n_workers = PipelineParameter().nThreads
        # n_workers = 10
        # we split the edges in 500 blocks
        n_splits = 500
        with futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            tasks = []
            for ii in xrange(n_splits):
                edge_start = int(float(ii) / n_splits * n_edges)
                edge_stop  = n_edges if ii == n_splits - 1 else int(float(ii + 1) / n_splits  * n_edges)
                tasks.append(executor.submit(
                    feats_for_subset,
                    uv_ids[edge_start:edge_stop, :],
                    edge_start)
                )
            [t.result() for t in tasks]

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs done.")

        if isinstance(skip_ranges, np.ndarray):
            assert skip_ranges.shape == (n_edges,)
            out.write([0, n_feats - 1], skip_ranges[:, None], key)

    def _compute_standard_feats(self, node_feats, inp, out):

        rag = inp['rag']
        uv_ids = rag.readKey('uvIds')
        transition_edge = rag.readKey('totalNumberOfInSliceEdges')

        # xy-feature
        self._compute_feats_from_uvs(node_feats, uv_ids[:transition_edge], "features_xy", out)
        # z-feature
        self._compute_feats_from_uvs(node_feats, uv_ids[transition_edge:], "features_z", out)

    # calculate and insert region features for the skip_edges
    # and delete the delete_edges
    def _compute_modified_feats(self, node_feats, inp, out):

        rag = inp['rag']
        modified_adjacency = inp['modified_adjacency']
        uv_ids = rag.readKey('uvIds')
        transition_edge = rag.readKey('totalNumberOfInSliceEdges')

        # compute the standard xy-features with additional ranges
        self._compute_feats_from_uvs(
            node_feats,
            uv_ids[:transition_edge],
            'features_xy',
            out,
            np.zeros(transition_edge, dtype='float32')
        )

        # compute the z-features with proper edges deleted from uv-ids
        delete_edges = modified_adjacency.read('delete_edges')
        uvs_z = uv_ids[transition_edge:]
        if delete_edges.size:
            assert delete_edges.min() >= transition_edge
            delete_edges -= transition_edge
            uvs_z = np.delete(uvs_z, delete_edges, axis=0)
        self._compute_feats_from_uvs(
            node_feats,
            uvs_z,
            'features_z',
            out,
            np.ones(uvs_z.shape[0], dtype='float32')
        )

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        assert skip_ranges.shape[0] == skip_edges.shape[0]

        # if we have skip edges, compute features for them
        if skip_edges.size:
            self._compute_feats_from_uvs(
                node_feats,
                skip_edges,
                'features_skip',
                out,
                skip_ranges
            )

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "RegionFeatures_")
        if PipelineParameter().defectPipeline:
            save_path += "modified_%s.h5" % seg_file
        else:
            save_path += "standard_%s.h5" % seg_file
        return HDF5VolumeTarget(save_path, 'float32')


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()
    keepOnlyXY = luigi.BoolParameter(default=False)
    keepOnlyZ = luigi.BoolParameter(default=False)

    # For now we can't set these any more, needs to be passed to C++ somehow
    # filterNames = luigi.ListParameter(
    #     default=[
    #         "gaussianSmoothing",
    #         "hessianOfGaussianEigenvalues",
    #         "laplacianOfGaussian"]
    # )
    # sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )

    def requires(self):
        required_tasks = {
            'rag': StackedRegionAdjacencyGraph(self.pathToSeg),
            'data': InputData(self.pathToInput)
        }
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):
        assert not(self.keepOnlyXY and self.keepOnlyZ)

        inp = self.input()
        rag = inp['rag'].read()
        data_file = inp['data']
        data_file.open()
        data = data_file.get()
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
        data_file.close()

        if PipelineParameter().defectPipeline:
            if modified_adjacency.read('has_defects'):
                self._postprocess_modified_output(out)

    def _compute_standard_feats(self, data, rag, out):

        workflow_logger.info("EdgeFeatures: _compute_standard_feats called.")
        n_edges_xy = rag.totalNumberOfInSliceEdges if not self.keepOnlyZ else 0
        n_edges_z  = rag.totalNumberOfInBetweenSliceEdges if not self.keepOnlyXY else 0

        # 9 * 12 = number of features per edge / would be nice not to hard code this here...
        n_feats = 9 * 12
        out_shape_xy    = [n_edges_xy, n_feats]
        chunk_shape_xy  = [min(2500, n_edges_xy), n_feats]
        out_shape_z    = [n_edges_z, n_feats]
        chunk_shape_z  = [min(2500, n_edges_z), n_feats]

        out.open(out_shape_xy, chunk_shape_xy, 'features_xy')
        out.open(out_shape_z, chunk_shape_z, 'features_z')

        nrag.accumulateEdgeFeaturesFromFilters(
            rag,
            data,
            out.get('features_xy'),
            out.get('features_z'),
            self.keepOnlyXY,
            self.keepOnlyZ,
            PipelineParameter().nThreads
        )
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
            chunk_shape = [min(2500, n_modified), n_feats]
            out.open(out_shape, chunk_shape, 'features_z_keep')

            # find all edges with continuous indices that will be deleted
            consecutive_deletes = np.split(
                delete_edges,
                np.where(np.diff(delete_edges) != 1)[0] + 1
            )

            prev_edge, total_copied = 0, 0
            # find the interval of edges to keep
            keep_edge_intervals = []
            if prev_edge != consecutive_deletes[0][0]:
                keep_edge_intervals.append([prev_edge, consecutive_deletes[0][0]])

            keep_edge_intervals.extend(
                [[consecutive_deletes[i][-1] + 1, consecutive_deletes[i + 1][0]]
                 for i in xrange(len(consecutive_deletes) - 1)]
            )

            if consecutive_deletes[-1][-1] != standard_feat_shape[0] - 1:
                keep_edge_intervals.append(
                    [consecutive_deletes[-1][-1] + 1, standard_feat_shape[0]]
                )

            for keep_start, keep_stop in keep_edge_intervals:
                n_copy = keep_stop - keep_start
                assert n_copy > 0, str(n_copy)
                out.write(
                    [total_copied, 0],
                    out.read([keep_start, 0], [keep_stop, n_feats], 'features_z'),
                    'features_z_keep'
                )
                total_copied += n_copy

            assert total_copied == standard_feat_shape[0] - delete_edges.shape[0], "%i, %i" % (
                total_copied,
                standard_feat_shape[0] - delete_edges.shape[0]
            )

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        skip_starts = modified_adjacency.read('skip_starts')
        assert skip_ranges.shape[0] == skip_edges.shape[0]
        assert skip_starts.shape[0] == skip_edges.shape[0]

        # modify the features only if we have skip edges
        if skip_edges.size:

            # TODO i/o in nifty to speed up calculation
            skip_feats = nrag.accumulateSkipEdgeFeaturesFromFilters(
                rag,
                data,
                # skip_edges need to be passed as a list of pairs!
                [(int(skip_e[0]), int(skip_e[1])) for skip_e in skip_edges],
                list(skip_ranges),
                list(skip_starts),
                PipelineParameter().nThreads
            )

            assert skip_feats.shape[0] == skip_edges.shape[0]
            assert skip_feats.shape[1] == n_feats
            print "Going to write skip edges"  # could use h5py too
            out.open(
                skip_feats.shape,
                [min(2500, skip_feats.shape[0]), skip_feats.shape[1]],
                'features_skip'
            )
            out.write([0, 0], skip_feats, 'features_skip')

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
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        inp_file = os.path.split(self.pathToInput)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "EdgeFeatures_%s_%s" % (seg_file, inp_file))
        if PipelineParameter().defectPipeline:
            save_path += '_modified'
        else:
            save_path += '_standard'
        if self.keepOnlyXY:
            save_path += '_xy'
        if self.keepOnlyZ:
            save_path += '_z'
        save_path += '.h5'
        return HDF5VolumeTarget(save_path, 'float32')


# TODO in nifty ??
# the edgeLens are implemented, rest will be more tricky and is not that helpful anyway...

# class TopologyFeatures(luigi.Task):
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
