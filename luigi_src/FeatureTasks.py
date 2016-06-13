# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi
from CustomTargets import HDF5Target
from PipelineParameter import *
from DataTasks import InputData, RegionAdjacencyGraph, ExternalSegmentationLabeled
from MiscTasks import EdgeIndications

import logging
import json

import os
import numpy as np
import vigra


# TODO
# class FilterSven
# proper feature hierarchies

class FilterVigra(luigi.Task):

    PathToInput = luigi.Parameter()

    FilterName = luigi.Parameter()
    Sigma = luigi.Parameter()
    Anisotropy = luigi.Parameter()

    def requires(self):
        return InputData(self.PathToInput)

    def run(self):

        inp = self.input().read()

        # TODO assert thtat this exists
        eval_filter = eval( ".".join( ["vigra", "filters", self.FilterName] ) )

        # calculate filter purely in 2d
        if self.Anisotropy > PipelineParameter().max_aniso:
            res = []
            for z in range(inp.shape[2]):
                filt_z = eval_filter( inp[:,:,z], sig )
                assert len(filt_z.shape) in (2,3)
                # insert z axis to stack later
                if len(filt_z.shape) == 2:
                    # single channel filter
                    filt_z = filt_z[:,:,np.newaxis]
                elif len(filt_z.shape) == 3:
                    # multi channel filter
                    filt_z = filt_z[:,:,np.newaxis,:]
                res.append(filt_z)
            # stack them together
            res = np.concatenate(res, axis = 2)
        else:
            if self.Anisotropy > 1.:
                sig = (self.Sigma, self.Sigma, self.Sigma / self.Anisotropy)
            else:
                sig = self.Sigma

            res = eval_filter( inp, sig )

        self.output().write(res)


    def output(self):
        aniso = self.Anisotropy
        if aniso > PipelineParameter().max_aniso:
            aniso = PipelineParameter().max_aniso
        return HDF5Target(
                os.path.join( PipelineParameter().cache, "_".join(
                    [os.path.split(self.PathToInput)[1], self.FilterName, str(self.Sigma), str(aniso)] ) + ".h5") )


# TODO svens filters, blockwise, chunked, presmoothing
# implement this as function, because we don't want to cache the filters!
def filter_vigra(PathToInput, FilterName, Sigma, Anisotropy):

    inp = vigra.readHDF5(PathToInput, "data")

    # TODO assert thtat this exists
    eval_filter = eval( ".".join( ["vigra", "filters", FilterName] ) )

    # calculate filter purely in 2d
    if Anisotropy > PipelineParameter().max_aniso:
        res = []
        for z in range(inp.shape[2]):
            filt_z = eval_filter( inp[:,:,z], Sigma )
            assert len(filt_z.shape) in (2,3)
            # insert z axis to stack later
            if len(filt_z.shape) == 2:
                # single channel filter
                filt_z = filt_z[:,:,np.newaxis]
            elif len(filt_z.shape) == 3:
                # multi channel filter
                filt_z = filt_z[:,:,np.newaxis,:]
            res.append(filt_z)
        # stack them together
        res = np.concatenate(res, axis = 2)
    else:
        if Anisotropy > 1.:
            sig = (Sigma, Sigma, Sigma / Anisotropy)
        else:
            sig = Sigma

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

    anisotropy  = feat_params["anisotropy"]
    filternames = feat_params["filternames"]
    sigmas = feat_params["sigmas"]
    features2d = feat_params["features2d"]

    # TODO check for invalid keys
    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append( EdgeFeatures(input_data[0], inputs["seg"],
                filternames, sigmas, anisotropy) )
    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeFeatures(input_data[1], inputs["seg"],
                filternames, sigmas, anisotropy) )
    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        # TODO should try it on probmaps. For big data we might spare shipping the raw data!
        feature_tasks.append( RegionFeatures(input_data[0], inputs["seg"]) )
    if "topo" in features:
        # by convention we calculate region features only on the raw data (0th input)
        feature_tasks.append( TopologyFeatures(inputs["seg"], features2d ) )

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

        inp = self.input()["Data"].read()
        seg = self.input()["Seg"].read()
        rag = self.input()["RAG"].read()

        uv_ids = np.sort( rag.uvIds(), axis = 1 )

        #if lifted_nh:
        #    print "Computing Lifted Region Features for NH:", lifted_nh
        #else:
        #    print "Computing Region features for local Edges"

        region_statistics, region_statistics_names = region_statistics(inp, seg)

        regStats = []

        for regStatName in region_statistics_names[0:10]:
            regStat = region_statistics[regStatName]
            if regStat.ndim == 1:
                regStats.append(regStat[:,None])
            else:
                regStats.append(regStat)

        regStats = np.concatenate(regStats, axis=1)

        regCenters = []
        for regStatName in  region_statistics_names[10:12]:
            regCenter = region_statistics[regStatName]
            if regCenter.ndim == 1:
                regCenters.append(regCenter[:,None])
            else:
                regCenters.append(regCenter)

        regCenters = np.concatenate(regCenters, axis=1)

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        del region_statistics
        gc.collect()

        fU = regStats[uv_ids[:,0],:]
        fV = regStats[uv_ids[:,1],:]

        allFeat = [np.minimum(fU, fV),
                np.maximum(fU, fV),
                np.abs(fU - fV),
                fU + fV ]

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        fV = fV.resize((1,1))
        fU = fU.resize((1,1))
        del fU
        del fV
        gc.collect()

        sU = regCenters[uv_ids[:,0],:]
        sV = regCenters[uv_ids[:,1],:]
        allFeat.append( (sU - sV)**2 )

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        sV = sV.resize((1,1))
        sU = sU.resize((1,1))
        del sU
        del sV
        gc.collect()

        allFeat = np.concatenate(allFeat, axis = 1)

        self.output().write( np.nan_to_num(allFeat) )


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

        self.output().write(topology_features)

    def output(self):
        return HDF5Target( os.path.join( PipelineParameter().cache, "TopologyFeatures.h5" ) )


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    PathToInput = luigi.Parameter()
    # current oversegmentation
    PathToSeg = luigi.Parameter()
    FilterNames = luigi.ListParameter(default = [ "vigra.filters.gaussianSmoothing", "vigra.filters.hessianOfGaussianEigenvalues", "vigra.filters.laplacianOfGaussian"] )
    Sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )
    Anisotropy = luigi.Parameter(default = 25.)

    def requires(self):
        return RegionAdjacencyGraph(self.PathToSeg), InputData(self.PathToInput)

    def run(self):
        edge_features = []
        rag = self.input()[0].read()
        for filter_name in self.FilterNames:
            for sigma in self.Sigmas:
                filt = filter_vigra(self.PathToInput, filter_name, sigma, self.Anisotropy)

                if len(filt.shape) == 3:
                    # let RAG do the work
                    gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, filt)
                    edge_features.append( rag.accumulateEdgeStatistics(gridGraphEdgeIndicator) )

                elif len(filt.shape) == 4:
                    for c in range(filt.shape[3]):
                        gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(
                                rag.baseGraph, filt[:,:,:,c] )
                        edge_features.append(rag.accumulateEdgeStatistics(gridGraphEdgeIndicator))

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == rag.edgeNum, str(edge_features.shape[0]) + " , " +str(rag.edgeNum)

        edge_features = np.nan_to_num(edge_features)

        self.output().write(edge_features)


    def output(self):
        inp_name = os.path.split(self.PathToInput)[1][:-3]
        seg_name = os.path.split(self.PathToSeg)[1][:-3]
        return HDF5Target( os.path.join( PipelineParameter().cache,
            "EdgeFeatures_" + inp_name + "_" + seg_name + ".h5" ) )
