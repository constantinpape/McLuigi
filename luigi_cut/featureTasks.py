# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from customTargets import HDF5Target
from dataTasks import InputData, InputDataChunked, RegionAdjacencyGraph, ExternalSegmentationLabeled
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

    # TODO check for invalid keys
    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append( EdgeFeatures(input_data[0], inputs["seg"]) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[0])
    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeFeatures(input_data[1], inputs["seg"] ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[1])
    #if "reg" in features:
    #    # by convention we calculate region features only on the raw data (0th input)
    #    # TODO should try it on probmaps. For big data we might spare shipping the raw data!
    #    feature_tasks.append( RegionFeatures(input_data[0], inputs["seg"]) )
    #    workflow_logger.debug("Calculating Region Features")
    #if "topo" in features:
    #    # by convention we calculate region features only on the raw data (0th input)
    #    feature_tasks.append( TopologyFeatures(inputs["seg"], features2d ) )
    #    workflow_logger.debug("Calculating Topology Features")

    return feature_tasks


# TODO implement in nifty
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

# TODO in nifty ??
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


# TODO adjust for nifty
class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()

    # For now we can't set these any more
    #filterNames = luigi.ListParameter(default = [ "gaussianSmoothing", "hessianOfGaussianEigenvalues", "laplacianOfGaussian"] )
    #sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )

    def requires(self):
        return StackedRegionAdjacencyGraph(self.PathToSeg), InputData(self.PathToInput)

    def run(self):

        rag = self.input()[0].read()
        self.input()[1].open()
        data = self.input()[1].get()

        t_feats = time.time()
        edge_features = np.nan_to_num( nifty.graph.rag.accumulateEdgeStatisticsFromFilters(rag, data) ) # nthreads
        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Edge Features in: " + str(t_feats) + " s")

        self.output().write(edge_features)


    def output(self):
        #inp_name = os.path.split(self.PathToInput)[1][:-3]
        #seg_name = os.path.split(self.PathToSeg)[1][:-3]
        #return HDF5Target( os.path.join( PipelineParameter().cache,
        #    "EdgeFeatures_" + inp_name + "_" + seg_name + ".h5" ) )
        return HDF5DataTarget( os.path.join( PipelineParameter().cache, "EdgeFeatures.h5" ) )
