# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from customTargets import HDF5Target
from dataTasks import InputData, RegionAdjacencyGraph, ExternalSegmentationLabeled
from miscTasks import EdgeIndications

from pipelineParameter import PipelineParameter
from toolsLuigi import config_logger

import logging
import json

import os
import time
import numpy as np
import vigra


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# we have this as a function now to calculate it on the fly
# maybe want to go back to a task once, so leave it in for now

#class FilterVigra(luigi.Task):
#
#    PathToInput = luigi.Parameter()
#
#    FilterName = luigi.Parameter()
#    Sigma = luigi.Parameter()
#    Anisotropy = luigi.Parameter()
#
#    def requires(self):
#        return InputData(self.PathToInput)
#
#    def run(self):
#
#        inp = self.input().read()
#
#        # TODO assert thtat this exists
#        eval_filter = eval( ".".join( ["vigra", "filters", self.FilterName] ) )
#
#        # calculate filter purely in 2d
#        if self.Anisotropy > PipelineParameter().max_aniso:
#            res = []
#            for z in range(inp.shape[2]):
#                filt_z = eval_filter( inp[:,:,z], sig )
#                assert len(filt_z.shape) in (2,3)
#                # insert z axis to stack later
#                if len(filt_z.shape) == 2:
#                    # single channel filter
#                    filt_z = filt_z[:,:,np.newaxis]
#                elif len(filt_z.shape) == 3:
#                    # multi channel filter
#                    filt_z = filt_z[:,:,np.newaxis,:]
#                res.append(filt_z)
#            # stack them together
#            res = np.concatenate(res, axis = 2)
#        else:
#            if self.Anisotropy > 1.:
#                sig = (self.Sigma, self.Sigma, self.Sigma / self.Anisotropy)
#            else:
#                sig = self.Sigma
#
#            res = eval_filter( inp, sig )
#
#        self.output().write(res)
#
#
#    def output(self):
#        aniso = self.Anisotropy
#        if aniso > PipelineParameter().max_aniso:
#            aniso = PipelineParameter().max_aniso
#        return HDF5Target(
#                os.path.join( PipelineParameter().cache, "_".join(
#                    [os.path.split(self.PathToInput)[1], self.FilterName, str(self.Sigma), str(aniso)] ) + ".h5") )


# TODO svens filters, blockwise, chunked, presmoothing
# implement this as function, because we don't want to cache the filters!
def calculate_filter(inp, filter_library, filter_name, sigma, anisotropy):

    assert filter_library in ("vigra", "fastfilters"), filter_library
    # svens filters
    if filter_library == "fastfilters":
        import fastfilters
        eval_filter = eval( ".".join( ["fastfilters", filter_name] ) )

    else:
        eval_filter = eval( ".".join( ["vigra", "filters", filter_name] ) )

    workflow_logger.debug("Calculating " + filter_name + " for anisotropy factor " + str(anisotropy))

    # calculate filter purely in 2d
    if anisotropy > PipelineParameter().MaxAniso:
        workflow_logger.debug("Filter calculation in 2d")

        # code not parallelized

        #res = []
        #for z in xrange(inp.shape[2]):
        #    filt_z = eval_filter( inp[:,:,z], sigma )
        #    assert len(filt_z.shape) in (2,3)
        #    # insert z axis to stack later
        #    if len(filt_z.shape) == 2:
        #        # single channel filter
        #        filt_z = filt_z[:,:,np.newaxis]
        #    elif len(filt_z.shape) == 3:
        #        # multi channel filter
        #        filt_z = filt_z[:,:,np.newaxis,:]
        #    res.append(filt_z)
        ## stack them together
        #res = np.concatenate(res, axis = 2)

        # code parallelized

        from concurrent import futures

        t_filt_pure = time.time()
        #with futures.ThreadPoolExecutor(max_workers = PipelineParameter().nThreads) as executor:
        with futures.ThreadPoolExecutor(max_workers = 8) as executor:
            tasks = []
            for z in xrange(inp.shape[2]):
                tasks.append( executor.submit(eval_filter, inp[:,:,z], sigma ) )
        workflow_logger.debug("Pure calculation time: " + str(time.time() - t_filt_pure))

        res = [task.result() for task in tasks]

        # TODO this is not really efficient !

        if res[0].ndim == 2:
            res = [re[:,:,None] for re in res]
        elif res[0].ndim == 3:
            res = [re[:,:,None,:] for re in res]

        res = np.concatenate( res, axis = 2)

    else:
        workflow_logger.debug("Filter calculation in 3d")
        if anisotropy > 1.:
            sig = (sigma, sigma, sigma / anisotropy)
        else:
            sig = sigma

        res = eval_filter( inp, sig )

    return res


# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features():
    # load the paths to input files
    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)
    # load the feature config
    with open(PipelineParameter().FeatureConfigFile, 'r') as f:
        feat_params = json.load(f)

    feature_tasks = []
    features = feat_params["features"]
    if not isinstance(features, list):
        features = [features,]

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data,]

    filter_library = feat_params["filter_library"]
    assert filter_library in ("vigra", "fastfilters"), filter_library
    anisotropy  = feat_params["anisotropy"]
    filternames = feat_params["filternames"]
    sigmas = feat_params["sigmas"]
    features2d = feat_params["features2d"]

    # TODO check for invalid keys
    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append( EdgeFeatures(input_data[0], inputs["seg"], filter_library,
                filternames, sigmas, anisotropy) )
        workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[0])
    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeFeatures(input_data[1], inputs["seg"], filter_library,
                filternames, sigmas, anisotropy) )
        workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[1])
    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        # TODO should try it on probmaps. For big data we might spare shipping the raw data!
        feature_tasks.append( RegionFeatures(input_data[0], inputs["seg"]) )
        workflow_logger.debug("Calculating Region Features")
    if "topo" in features:
        # by convention we calculate region features only on the raw data (0th input)
        feature_tasks.append( TopologyFeatures(inputs["seg"], features2d ) )
        workflow_logger.debug("Calculating Topology Features")

    return feature_tasks


#def region_features(self, seg_id, inp_id, uv_ids, lifted_nh):
class RegionFeatures(luigi.Task):

    PathToInput = luigi.Parameter()
    PathToSeg   = luigi.Parameter()

    def requires(self):
        # TODO have to rethink this once we include lifted multicut
        return {"Data" : InputData(self.PathToInput), "Seg" : ExternalSegmentationLabeled(self.PathToSeg), "RAG" : RegionAdjacencyGraph(self.PathToSeg)}


    def run(self):

        # get region statistics with the vigra region feature extractor
        def region_statistics(inp, seg):
            # list of the region statistics, that we want to extract
            statistics =  [ "Histogram", "Count", "Kurtosis",
                            "Maximum", "Minimum", "Quantiles",
                            "RegionRadii", "Skewness", "Sum",
                            "Variance", "Weighted<RegionCenter>", "RegionCenter"]

            extractor = vigra.analysis.extractRegionFeatures(
                    inp.astype(np.float32),
                    seg.astype(np.uint32),
                    features = statistics )

            return extractor, statistics

        import gc

        t_feats = time.time()

        inp = self.input()["Data"].read()
        seg = self.input()["Seg"].read()
        rag = self.input()["RAG"].read()

        uv_ids = np.sort( rag.uvIds(), axis = 1 )

        #if lifted_nh:
        #    print "Computing Lifted Region Features for NH:", lifted_nh
        #else:
        #    print "Computing Region features for local Edges"

        region_statistics, region_statistics_names = region_statistics(inp, seg)

        reg_stats = []

        for reg_stat_name in region_statistics_names[0:10]:
            reg_stat = region_statistics[reg_stat_name]
            if reg_stat.ndim == 1:
                reg_stats.append(reg_stat[:,None])
            else:
                reg_stats.append(reg_stat)

        reg_stats = np.concatenate(reg_stats, axis=1)

        reg_centers = []
        for reg_stat_name in region_statistics_names[10:12]:
            reg_center = region_statistics[reg_stat_name]
            if reg_center.ndim == 1:
                reg_centers.append(reg_center[:,None])
            else:
                reg_centers.append(reg_center)

        reg_centers = np.concatenate(reg_centers, axis=1)

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        del region_statistics
        gc.collect()

        f_u = reg_stats[uv_ids[:,0],:]
        f_v = reg_stats[uv_ids[:,1],:]

        all_feat = [np.minimum(f_u, f_v), np.maximum(f_u, f_v),
                np.abs(f_u - f_v), f_u + f_v ]

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        f_u = f_u.resize((1,1))
        f_v = f_v.resize((1,1))
        del f_u
        del f_v
        gc.collect()

        s_u = reg_centers[uv_ids[:,0],:]
        s_v = reg_centers[uv_ids[:,1],:]
        all_feat.append( (s_u - s_v)**2 )

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        s_u = s_u.resize((1,1))
        s_v = s_v.resize((1,1))
        del s_u
        del s_v
        gc.collect()

        all_feat = np.concatenate(all_feat, axis = 1)

        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Region Features in: " + str(t_feats) + " s")

        self.output().write( np.nan_to_num(all_feat) )


    def output(self):
        return HDF5Target( os.path.join(
            PipelineParameter().cache, "RegionFeatures.h5" ) )


class TopologyFeatures(luigi.Task):

    PathToSeg = luigi.Parameter()
    Use2dFeatures = luigi.BoolParameter(default = True)

    def requires(self):
        if self.Use2dFeatures:
            return {"Seg" : ExternalSegmentationLabeled(self.PathToSeg) , "RAG" : RegionAdjacencyGraph(self.PathToSeg),
                    "EdgeIndications" : EdgeIndications(self.PathToSeg) }
        else:
            return {"Seg" : ExternalSegmentationLabeled(self.PathToSeg) , "RAG" : RegionAdjacencyGraph(self.PathToSeg)}

    # Features from edge_topology
    #def topology_features(self, seg_id, use_2d_edges):
    def run(self):

        t_feats = time.time()

        rag = self.input()["RAG"].read()
        seg = self.input()["Seg"].read()

        if self.Use2dFeatures:
            n_feats = 7
        else:
            n_feats = 1

        n_edges = rag.edgeNum
        topology_features = np.zeros( (n_edges, n_feats) )

        # length / area of the edge
        edge_lens = rag.edgeLengths()
        assert edge_lens.shape[0] == n_edges
        topology_features[:,0] = edge_lens

        # extra feats for z-edges in 2,5 d
        if self.Use2dFeatures:

            # edge indications
            edge_indications = self.input()["EdgeIndications"].read()
            assert edge_indications.shape[0] == n_edges
            topology_features[:,1] = edge_indications

            # region sizes to build some features
            statistics =  [ "Count", "RegionCenter" ]

            extractor = vigra.analysis.extractRegionFeatures(
                    np.zeros_like(seg, dtype = np.float32), # dummy input
                    seg, features = statistics )

            z_mask = edge_indications == 0

            sizes = extractor["Count"]
            uvIds = np.sort( rag.uvIds(), axis = 1)
            sizes_u = sizes[ uvIds[:,0] ]
            sizes_v = sizes[ uvIds[:,1] ]
            # union = size_up + size_dn - intersect
            unions  = sizes_u + sizes_v - edge_lens
            # Union features
            topology_features[:,2][z_mask] = unions[z_mask]
            # IoU features
            topology_features[:,3][z_mask] = edge_lens[z_mask] / unions[z_mask]

            # segment shape features
            seg_coordinates = extractor["RegionCenter"]
            len_bounds      = np.zeros(rag.nodeNum)
            # TODO no loop ?! or CPP
            # iterate over the nodes, to get the boundary length of each node
            for n in rag.nodeIter():
                node_z = seg_coordinates[n.id][2]
                for arc in rag.incEdgeIter(n):
                    edge = rag.edgeFromArc(arc)
                    edge_c = rag.edgeCoordinates(edge)
                    # only edges in the same slice!
                    if edge_c[0,2] == node_z:
                        len_bounds[n.id] += edge_lens[edge.id]
            # shape feature = Area / Circumference
            shape_feats_u = sizes_u / len_bounds[uvIds[:,0]]
            shape_feats_v = sizes_v / len_bounds[uvIds[:,1]]
            # combine w/ min, max, absdiff
            print shape_feats_u[z_mask].shape
            print shape_feats_v[z_mask].shape
            topology_features[:,4][z_mask] = np.minimum(
                    shape_feats_u[z_mask], shape_feats_v[z_mask])
            topology_features[:,5][z_mask] = np.maximum(
                    shape_feats_u[z_mask], shape_feats_v[z_mask])
            topology_features[:,6][z_mask] = np.absolute(
                    shape_feats_u[z_mask] - shape_feats_v[z_mask])

        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Topology Features in: " + str(t_feats) + " s")

        self.output().write(topology_features)

    def output(self):
        return HDF5Target( os.path.join( PipelineParameter().cache, "TopologyFeatures.h5" ) )


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    PathToInput = luigi.Parameter()
    # current oversegmentation
    PathToSeg = luigi.Parameter()
    FilterLibrary = luigi.Parameter(default = "vigra")
    FilterNames = luigi.ListParameter(default = [ "vigra.filters.gaussianSmoothing", "vigra.filters.hessianOfGaussianEigenvalues", "vigra.filters.laplacianOfGaussian"] )
    Sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )
    Anisotropy = luigi.Parameter(default = 25.)

    def requires(self):
        return RegionAdjacencyGraph(self.PathToSeg), InputData(self.PathToInput)

    def run(self):

        t_feats = time.time()

        edge_features = []
        rag = self.input()[0].read()
        inp = self.input()[1].read()

        for filter_name in self.FilterNames:
            for sigma in self.Sigmas:
                t_filt = time.time()
                workflow_logger.info("Calculation of " + filter_name + " for sigma: " + str(sigma)  )
                filt = calculate_filter(inp, self.FilterLibrary, filter_name, sigma, self.Anisotropy)
                workflow_logger.info("Filter calculation in "+ str( time.time() - t_filt))

                t_acc = time.time()
                if len(filt.shape) == 3:
                    # let RAG do the work
                    grid_graph_edge_indicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, filt)
                    edge_features.append( rag.accumulateEdgeStatistics( grid_graph_edge_indicator) )

                elif len(filt.shape) == 4:
                    for c in range(filt.shape[3]):
                        grid_graph_edge_indicator = vigra.graphs.implicitMeanEdgeMap(
                                rag.baseGraph, filt[:,:,:,c] )
                        edge_features.append(rag.accumulateEdgeStatistics(grid_graph_edge_indicator))

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == rag.edgeNum, str(edge_features.shape[0]) + " , " +str(rag.edgeNum)
        workflow_logger.info("Accumulation over the edges in " + str(time.time() - t_acc))

        edge_features = np.nan_to_num(edge_features)

        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Edge Features in: " + str(t_feats) + " s")

        self.output().write(edge_features)


    def output(self):
        #inp_name = os.path.split(self.PathToInput)[1][:-3]
        #seg_name = os.path.split(self.PathToSeg)[1][:-3]
        #return HDF5Target( os.path.join( PipelineParameter().cache,
        #    "EdgeFeatures_" + inp_name + "_" + seg_name + ".h5" ) )
        return HDF5Target( os.path.join( PipelineParameter().cache, "EdgeFeatures.h5" ) )
