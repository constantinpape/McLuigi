from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from .customTargets import VolumeTarget, HDF5DataTarget
from .dataTasks import InputData, StackedRegionAdjacencyGraph, ExternalSegmentation
from .defectHandlingTasks import ModifiedAdjacency
from .pipelineParameter import PipelineParameter
from .tools import config_logger, run_decorator

import logging
import os
import numpy as np
import vigra

from concurrent import futures

# import the proper nifty version
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi.graph.rag as nrag


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class RegionNodeFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()
    keyToInput  = luigi.Parameter(default='data')
    keyToSeg    = luigi.Parameter(default='data')

    def requires(self):
        return {"data": InputData(self.pathToInput),
                "seg": ExternalSegmentation(self.pathToSeg),
                "rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg)}

    @run_decorator
    def run(self):
        inp = self.input()
        data = inp["data"]
        seg  = inp["seg"]

        data.open(self.keyToInput)
        seg.open(self.keyToSeg)
        shape = data.shape(self.keyToInput)

        assert shape == seg.shape(self.keyToSeg), str(shape) + " , " + str(seg.shape())

        min_max_node = inp['rag'].readKey('minMaxLabelPerSlice').astype('uint32')
        n_nodes = inp['rag'].readKey('numberOfNodes')
        n_feats = 20

        # list of the region statistics, that we want to extract
        # drop te Histogram, because it blows up the feature space...
        # TODO also use Mean and add Histogram if needed
        statistics = ["Count", "Kurtosis",  # Histogram
                      "Maximum", "Minimum", "Quantiles",
                      "RegionRadii", "Skewness", "Sum",
                      "Variance", "Weighted<RegionCenter>", "RegionCenter"]

        out = self.output()
        out_shape = (n_nodes, n_feats)
        chunk_shape = (min(5000, n_nodes), out_shape[1])
        out.open(key="data", shape=out_shape, chunks=chunk_shape, dtype='float32')

        # get region statistics with the vigra region feature extractor for a single slice
        def extract_stats_slice(z):
            print("!!!!", z, "!!!!")
            start, end = [z, 0, 0], [z + 1, shape[1], shape[2]]
            min_node, max_node = min_max_node[z, 0], min_max_node[z, 1]

            data_slice = data.read(start, end, self.keyToInput).squeeze().astype('float32',
                                                                                 copy=False)
            seg_slice  = seg.read(start, end, self.keyToSeg).squeeze() - min_node

            print("Here")
            print(vigra.__file__)
            print(vigra.analysis.__file__)
            # FIXME some numpy issues...
            extractor = vigra.analysis.extractRegionFeatures(data_slice,
                                                             seg_slice.astype('uint32',
                                                                              copy=False),
                                                             features=statistics)
            region_stats_slice = []
            for stat_name in statistics:
                stat = extractor[stat_name]
                if stat.ndim == 1:
                    region_stats_slice.append(stat[:, None])
                else:
                    region_stats_slice.append(stat)
            region_stats_slice = np.nan_to_num(np.concatenate(region_stats_slice,
                                                              axis=1).astype('float32',
                                                                             copy=False))
            assert region_stats_slice.shape[0] == max_node + 1 - min_node
            out.writeSubarray((min_node, 0), region_stats_slice)

        # parallel
        # n_workers = min(shape[0], PipelineParameter().nThreads)
        n_workers = 1
        with futures.ThreadPoolExecutor(max_workers=n_workers) as tp:
            tasks = [tp.submit(extract_stats_slice, z) for z in range(shape[0])]
            [task.result() for task in tasks]
        out.close()

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "RegionNodeFeatures_%s" % seg_file)
        # save_path += VolumeTarget.file_ending()
        # return VolumeTarget(save_path)
        save_path += '.h5'
        return HDF5DataTarget(save_path)


# FIXME need to adjust chunks for parallel n5 writing
class RegionFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()
    keyToInput  = luigi.Parameter(default='data')
    keyToSeg    = luigi.Parameter(default='data')

    # TODO have to rethink this if we include lifted multicut
    def requires(self):
        required_tasks = {"rag": StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                          "node_feats": RegionNodeFeatures(pathToInput=self.pathToInput,
                                                           pathToSeg=self.pathToSeg,
                                                           keyToInput=self.keyToInput,
                                                           keyToSeg=self.keyToSeg)}
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):

        inp = self.input()
        out = self.output()

        node_feats = inp["node_feats"].read()
        if PipelineParameter().defectPipeline:
            modified_adjacency = inp['modified_adjacency']
            if modified_adjacency.read('has_defects'):
                self._compute_modified_feats(node_feats, inp, out)
            else:
                self._compute_standard_feats(node_feats, inp, out)
        else:
            self._compute_standard_feats(node_feats, inp, out)

        out.close()

    def _compute_feats_from_uvs(self,
                                node_feats,
                                uv_ids,
                                key,
                                out,
                                skip_ranges=None):

        if not isinstance(skip_ranges, np.ndarray):
            assert skip_ranges is None

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs called with key: %s" % key)
        n_edges = uv_ids.shape[0]
        # magic 16 = number of regionStatistics that are combined by min, max, sum and absdiff
        nStatFeats = 16

        n_feats = 4 * nStatFeats + 4
        if isinstance(skip_ranges, np.ndarray):
            n_feats += 1

        # we open the out file for this features
        out_shape = (n_edges, n_feats)
        chunk_shape = (2500, out_shape[1])
        out.open(key, dtype='float32', shape=out_shape, chunks=chunk_shape)

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
            out.write((edge_offset, 0), feats_sub, key)
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
            for ii in range(n_splits):
                edge_start = int(float(ii) / n_splits * n_edges)
                edge_stop  = n_edges if ii == n_splits - 1 else int(float(ii + 1) / n_splits  * n_edges)
                tasks.append(executor.submit(feats_for_subset,
                                             uv_ids[edge_start:edge_stop, :],
                                             edge_start))
            [t.result() for t in tasks]

        workflow_logger.info("RegionFeatures: _compute_feats_from_uvs done.")

        if isinstance(skip_ranges, np.ndarray):
            assert skip_ranges.shape == (n_edges,)
            out.writeSubarray((0, n_feats - 1), skip_ranges[:, None], key)

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
        self._compute_feats_from_uvs(node_feats, uv_ids[:transition_edge], 'features_xy', out)

        # compute the z-features with proper edges deleted from uv-ids
        delete_edges = modified_adjacency.read('delete_edges')
        uvs_z = uv_ids[transition_edge:]
        if delete_edges.size:
            assert delete_edges.min() >= transition_edge
            delete_edges -= transition_edge
            uvs_z = np.delete(uvs_z, delete_edges, axis=0)

        self._compute_feats_from_uvs(node_feats, uvs_z, 'features_z', out)

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        assert skip_ranges.shape[0] == skip_edges.shape[0]

        # if we have skip edges, compute features for them
        if skip_edges.size:
            self._compute_feats_from_uvs(node_feats,
                                         skip_edges,
                                         'features_skip',
                                         out,
                                         skip_ranges)

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join(PipelineParameter().cache, "RegionFeatures_")
        if PipelineParameter().defectPipeline:
            save_path += "modified_%s" % seg_file
        else:
            save_path += "standard_%s" % seg_file
        save_path += VolumeTarget.file_ending()
        return VolumeTarget(save_path)


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    keyToInput = luigi.Parameter(default='data')
    # current oversegmentation
    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default='data')

    # optional parameters
    keepOnlyXY = luigi.BoolParameter(default=False)
    keepOnlyZ = luigi.BoolParameter(default=False)
    simpleFeatures = luigi.BoolParameter(default=False)
    zDirection = luigi.IntParameter(default=0)

    # For now we can't set these any more, needs to be passed to C++ somehow
    # filterNames = luigi.ListParameter(
    #     default=[
    #         "gaussianSmoothing",
    #         "hessianOfGaussianEigenvalues",
    #         "laplacianOfGaussian"]
    # )
    # sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )

    def requires(self):
        required_tasks = {'rag': StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg),
                          'data': InputData(self.pathToInput)}
        if PipelineParameter().defectPipeline:
            required_tasks['modified_adjacency'] = ModifiedAdjacency(self.pathToSeg)
        return required_tasks

    @run_decorator
    def run(self):
        assert not(self.keepOnlyXY and self.keepOnlyZ)

        inp = self.input()
        rag = inp['rag'].read()
        data_file = inp['data']
        data_file.open(self.keyToInput)
        data = data_file.get(self.keyToInput)

        out = self.output()

        has_defects = False
        if PipelineParameter().defectPipeline:
            modified_adjacency = inp['modified_adjacency']
            if modified_adjacency.read('has_defects'):
                has_defects = True

        if has_defects:
            self._compute_modified_feats(data, rag, modified_adjacency, out)
        else:
            self._compute_standard_feats(data, rag, out)

        out.close()
        data_file.close()
        if PipelineParameter().useN5Backend:
            self._postprocess_output_n5(out, has_defects)
        else:
            self._postprocess_output_h5(out, has_defects)

    # we delete the old features_z and then rename the keep features
    def _postprocess_output_n5(self, out, has_defects):
        from shutil import move, rmtree
        if has_defects:
            z_path = os.path.join(out.path, 'features_z')
            rmtree(z_path)
            z_path_new = os.path.join(out.path, 'features_z_new')
            move(z_path_new, z_path)
        # if we only compute features for one of the edge-types
        # remove the features of the other type
        if self.keepOnlyXY:
            rmtree(os.path.join(out.path, 'features_z'))
        if self.keepOnlyZ:
            rmtree(os.path.join(out.path, 'features_xy'))

    def _postprocess_output_h5(self, out, has_defects):
        import h5py
        if has_defects:
            with h5py.File(out.path) as f:
                f['features_z_new'] = f['features_z']
                del f['features_z_new']
        # if we only compute features for one of the edge-types
        # remove the features of the other type
        if self.keepOnlyXY:
            with h5py.File(out.path) as f:
                del f['features_z']
        if self.keepOnlyZ:
            with h5py.File(out.path) as f:
                del f['features_xy']

    def _compute_standard_feats(self, data, rag, out):

        workflow_logger.info("EdgeFeatures: _compute_standard_feats called.")
        n_edges_xy = rag.totalNumberOfInSliceEdges if not self.keepOnlyZ else 1
        n_edges_z  = rag.totalNumberOfInBetweenSliceEdges if not self.keepOnlyXY else 1

        # as minimum chunk size, we choose the minimum number of edges
        # of a given type per slice
        min_edges_xy = min(np.min(rag.numberOfInSliceEdges()), n_edges_xy)
        min_edges_z = min(np.min(rag.numberOfInBetweenSliceEdges()[:-1]), n_edges_z)

        # number of features:
        # 9 * 12 for features from filter accumulation
        # 9 for simple features
        # TODO would be nice not to hard code this here...
        n_feats = 9 if self.simpleFeatures else 9 * 12

        # max chunk size s.t. n_feats * max_chunk_size ~ 64**3
        max_chunk_size = 30000 if self.simpleFeatures else 2500

        out_shape_xy    = (n_edges_xy, n_feats)
        # we choose the min in-slice edge number as minimum chunk size
        chunk_shape_xy  = (min(max_chunk_size, min_edges_xy), n_feats)

        out_shape_z    = (n_edges_z, n_feats)
        chunk_shape_z  = (min(max_chunk_size, min_edges_z), n_feats)

        # open the output files
        out.open('features_xy', dtype='float32', shape=out_shape_xy, chunks=chunk_shape_xy)
        out.open('features_z', dtype='float32', shape=out_shape_z, chunks=chunk_shape_z)

        if self.simpleFeatures:
            workflow_logger.info("EdgeFeatures: computing standard features.")
            nrag.accumulateEdgeStandardFeatures(rag, data,
                                                out.get('features_xy'), out.get('features_z'),
                                                self.keepOnlyXY, self.keepOnlyZ,
                                                self.zDirection,
                                                PipelineParameter().nThreads)
        else:
            workflow_logger.info("EdgeFeatures: computing features from filers.")
            nrag.accumulateEdgeFeaturesFromFilters(rag, data,
                                                   out.get('features_xy'), out.get('features_z'),
                                                   self.keepOnlyXY, self.keepOnlyZ,
                                                   self.zDirection,
                                                   PipelineParameter().nThreads)

        workflow_logger.info("EdgeFeatures: _compute_standard_feats done.")

    # TODO implement simpler feature computation in nifty
    def _compute_modified_feats(self, data, rag, modified_adjacency, out):

        workflow_logger.info("EdgeFeatures: _compute_modified_feats called.")
        # first, compute the standard feats
        self._compute_standard_feats(data, rag, out)

        transition_edge = rag.totalNumberOfInSliceEdges
        # copy the z-features we keep and delete the ones that are not needed
        delete_edges = modified_adjacency.read('delete_edges')

        has_delete_edges = delete_edges.size and not self.keepOnlyXY
        if has_delete_edges:
            assert delete_edges.min() >= transition_edge
            # we substract the transition edge, because we count from the begin of z edges
            delete_edges -= transition_edge

            # read the original z-features
            standard_feat_shape = out['features_z'].shape()

            n_modified = standard_feat_shape[0] - delete_edges.shape[0]
            n_feats    = standard_feat_shape[1]

            # open a new file for the modified edges
            out_shape   = (n_modified, n_feats)
            chunk_shape = (min(2500, n_modified), n_feats)
            out.open('features_z_new', dtype='float32', shape=out_shape, chunks=chunk_shape)

            # find all edges with continuous indices that will be deleted
            consecutive_deletes = np.split(delete_edges,
                                           np.where(np.diff(delete_edges) != 1)[0] + 1)
            prev_edge, total_copied = 0, 0
            # find the interval of edges to keep
            keep_edge_intervals = []
            if prev_edge != consecutive_deletes[0][0]:
                keep_edge_intervals.append([prev_edge, consecutive_deletes[0][0]])

            keep_edge_intervals.extend([[consecutive_deletes[i][-1] + 1, consecutive_deletes[i + 1][0]]
                                       for i in range(len(consecutive_deletes) - 1)])

            if consecutive_deletes[-1][-1] != standard_feat_shape[0] - 1:
                keep_edge_intervals.append([consecutive_deletes[-1][-1] + 1, standard_feat_shape[0]])

            for keep_start, keep_stop in keep_edge_intervals:
                n_copy = keep_stop - keep_start
                assert n_copy > 0, str(n_copy)
                out.write([total_copied, 0],
                          out.readSubarray([keep_start, 0], [keep_stop, n_feats], 'features_z'),
                          'features_z_new')
                total_copied += n_copy

            assert total_copied == standard_feat_shape[0] - delete_edges.shape[0], "%i, %i" % (
                total_copied,
                standard_feat_shape[0] - delete_edges.shape[0])

        skip_edges = modified_adjacency.read('skip_edges')
        skip_ranges = modified_adjacency.read('skip_ranges')
        skip_starts = modified_adjacency.read('skip_starts')
        assert skip_ranges.shape[0] == skip_edges.shape[0]
        assert skip_starts.shape[0] == skip_edges.shape[0]

        # modify the features only if we have skip edges
        if skip_edges.size and not self.keepOnlyXY:

            # TODO simple feats for skip features
            # TODO i/o in nifty to speed up calculation
            skip_feats = nrag.accumulateSkipEdgeFeaturesFromFilters(rag,
                                                                    data,
                                                                    # skip_edges need to be passed as a list of pairs!
                                                                    [(int(skip_e[0]), int(skip_e[1]))
                                                                     for skip_e in skip_edges],
                                                                    list(skip_ranges),
                                                                    list(skip_starts),
                                                                    self.zDirection,
                                                                    PipelineParameter().nThreads)

            assert skip_feats.shape[0] == skip_edges.shape[0]
            # TODO reactivate check once we have simple skip feats
            # assert skip_feats.shape[1] == n_feats, "%i, %i" % (skip_feats.shape[1], n_feats)

            # open file for the skip edges
            vigra.writeHDF5(skip_feats,
                            os.path.join(out.path, 'features_skip.h5'), 'data',
                            chunks=(min(2500, skip_feats.shape[0]), skip_feats.shape[1]))

    def output(self):
        seg_file = os.path.split(self.pathToSeg)[1][:-3]
        inp_file = os.path.split(self.pathToInput)[1][:-3]
        with_defects = PipelineParameter().defectPipeline
        save_path = os.path.join(PipelineParameter().cache,
                                 "EdgeFeatures_%s_%s_%s" % (seg_file,
                                                            inp_file,
                                                            'modified' if with_defects
                                                            else 'standard'))
        if self.keepOnlyXY:
            save_path += '_xy'
        if self.keepOnlyZ:
            save_path += '_z'
        if self.simpleFeatures:
            save_path += '_simple'
        if self.zDirection != 0:
            save_path += '_zDir%i' % self.zDirection
        save_path += VolumeTarget.file_ending()
        return VolumeTarget(save_path)
