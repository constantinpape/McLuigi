# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi

from customTargets import HDF5DataTarget
from dataTasks import InputData, StackedRegionAdjacencyGraph, ExternalSegmentation
#from miscTasks import EdgeIndications

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


# try vigra features for debugging
#from debugTasks import get_local_vigra_features
#get_local_features = get_local_vigra_features

# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features(xyOnly = False, zOnly = False):

    assert not (xyOnly and zOnly)

    # load the paths to input files
    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)
    # load the feature config
    with open(PipelineParameter().FeatureConfigFile, 'r') as f:
        feat_params = json.load(f)

    features = feat_params["features"]

    if not isinstance(features, list):
        features = [features,]

    feature_tasks = []

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data,]

    # TODO check for invalid keys
    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append( EdgeFeatures(input_data[0], inputs["seg"]) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[0])

    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeFeatures(input_data[1], inputs["seg"] ) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[1])

    if "affinitiesXY" in features and not zOnly: # specific XY - features -> we keep only these
        # by convention we assume that the xy - affinity channel is given as 1st input
        feature_tasks.append( EdgeFeatures(input_data[1], inputs["seg"], keepOnlyXY = True ) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from xy affinity maps: " + input_data[1])

    if "affinitiesZ" in features and not xyOnly: # specific Z - features -> we keep only these
        # by convention we assume that the z - affinity channel is given as 2nd input
        feature_tasks.append( EdgeFeatures(input_data[2], inputs["seg"], keepOnlyZ = True ) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from z affinity maps: " + input_data[2])

    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        # TODO should try it on probmaps. For big data we might spare shipping the raw data!
        feature_tasks.append( RegionFeatures(input_data[0], inputs["seg"]) )
        workflow_logger.debug("Calculating Region Features")

    #if "topo" in features:
    #    # by convention we calculate region features only on the raw data (0th input)
    #    feature_tasks.append( TopologyFeatures(inputs["seg"], features2d ) )
    #    workflow_logger.debug("Calculating Topology Features")

    return feature_tasks


# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features_for_multiinp(xyOnly = False, zOnly = False):

    assert not (xyOnly and zOnly)

    # load the paths to input files
    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)
    # load the feature config
    with open(PipelineParameter().FeatureConfigFile, 'r') as f:
        feat_params = json.load(f)

    features = feat_params["features"]
    if not isinstance(features, list):
        features = [features,]

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data,]

    segs = inputs["seg"]

    nInpPerSeg = len(input_data) / len(segs)

    feature_tasks = []
    for i in xrange(len(segs)):
        inp0 = nInpPerSeg*i
        inp1 = nInpPerSeg*i + 1
        inp2 = nInpPerSeg*i + 2

        feature_tasks.append([])

        if "raw" in features:
            # by convention we assume that the raw data is given as 0th
            feature_tasks[i].append( EdgeFeatures(input_data[inp0], segs[i]) ) #, filternames, sigmas) )
            workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[inp0])

        if "prob" in features:
            #assert nInpPerSeg == 2
            # by convention we assume that the membrane probs are given as 1st
            feature_tasks[i].append( EdgeFeatures(input_data[inp1], segs[i] ) ) #, filternames, sigmas) )
            workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[inp1])

        if "affinitiesXY" in features and not zOnly:
            assert nInpPerSeg == 3
            # by convention we assume that the xy - affinity channel is given as 1st input
            feature_tasks[i].append( EdgeFeatures(input_data[inp1], segs[i], keepOnlyXY = True ) ) #, filternames, sigmas) )
            workflow_logger.debug("Calculating Edge Features from xy affinity maps: " + input_data[inp1])

        if "affinitiesZ" in features and not xyOnly:
            assert nInpPerSeg == 3
            # by convention we assume that the z - affinity channel is given as 2nd input
            feature_tasks[i].append( EdgeFeatures(input_data[inp2], segs[i], keepOnlyZ = True ) ) #, filternames, sigmas) )
            workflow_logger.debug("Calculating Edge Features from z affinity maps: " + input_data[inp2])

        if "reg" in features:
            feature_tasks[i].append( RegionFeatures(input_data[inp0], segs[i]) )
            workflow_logger.debug("Calculating Region Features")

    return feature_tasks


class RegionFeatures(luigi.Task):

    pathToInput = luigi.Parameter()
    pathToSeg   = luigi.Parameter()

    def requires(self):
        # TODO have to rethink this once we include lifted multicut
        return {"Data" : InputData(self.pathToInput), "Seg" : ExternalSegmentation(self.pathToSeg), "Rag" : StackedRegionAdjacencyGraph(self.pathToSeg)}


    def run(self):

        import gc

        t_feats = time.time()

        inp = self.input()

        data = inp["Data"]
        seg  = inp["Seg"]
        rag  = inp["Rag"].read()

        data.open()
        seg.open()

        shape = data.shape

        assert data.shape == seg.shape, str(data.shape) + " , " + str(seg.shape)

        minMaxNodeSlice = rag.minMaxLabelPerSlice().astype('uint32')
        nNodes = rag.numberOfNodes

        # list of the region statistics, that we want to extract
        # drop te Histogram, because it blows up the feature space...
        statistics =  [ "Count", "Kurtosis", #Histogram
                        "Maximum", "Minimum", "Quantiles",
                        "RegionRadii", "Skewness", "Sum",
                        "Variance", "Weighted<RegionCenter>", "RegionCenter"]


        # get region statistics with the vigra region feature extractor for a single slice
        def extractRegionStatsSlice(start, end, z):

            minNode = minMaxNodeSlice[z,0]

            dataSlice = data.read(start,end).squeeze().astype('float32')
            segSlice  = seg.read(start,end).squeeze() - minNode

            extractor = vigra.analysis.extractRegionFeatures(dataSlice, segSlice, features = statistics )

            regionStatistics = []
            regionCenters    = []

            # the 'normal' region statistics
            for statName in statistics[0:9]:
                # only keep the nodes that are in this slice (others are zero anyway!)
                stat = extractor[statName]
                if stat.ndim == 1:
                    regionStatistics.append(stat[:,None])
                else:
                    regionStatistics.append(stat)

            regionStatistics = np.concatenate(regionStatistics, axis = 1)

            # the region center differences, that art treated seperately
            for statName in statistics[9:11]:
                # only keep the nodes that are in this slice (others are zero anyway!)
                stat = extractor[statName]
                if stat.ndim == 1:
                    regionCenters.append(stat[:,None])
                else:
                    regionCenters.append(stat)

            regionCenters = np.concatenate(regionCenters, axis=1)

            return (regionStatistics, regionCenters)


        nWorkers = min( shape[0], PipelineParameter().nThreads )
        #nWorkers = 1
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for z in xrange(shape[0]):
                start = [z,0,0]
                end   = [z+1,shape[1],shape[2]]
                tasks.append( executor.submit(extractRegionStatsSlice, start, end, z) )

        results = [task.result() for task in tasks]

        regionStatistics = np.zeros( (nNodes, results[0][0].shape[1]) )
        regionCenters    = np.zeros( (nNodes, results[0][1].shape[1]) )

        # TODO should be vectorized or also done in parallel
        for z, res in enumerate(results):
            minNode    = minMaxNodeSlice[z,0]
            maxNode    = minMaxNodeSlice[z,1]
            regionStatistics[minNode:maxNode+1] = res[0]
            regionCenters[minNode:maxNode+1] = res[1]

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        results = []
        del results
        gc.collect()

        uvIds = rag.uvIds()

        fU = regionStatistics[uvIds[:,0],:]
        fV = regionStatistics[uvIds[:,1],:]

        regionStatistics.resize((1,1))
        del regionStatistics
        gc.collect()

        allFeats = [
                np.minimum(fU, fV),
                np.maximum(fU, fV),
                np.abs(fU - fV),
                fU + fV
                ]

        fU = fU.resize((1,1))
        fV = fV.resize((1,1))
        del fU
        del fV
        gc.collect()

        sU = regionCenters[uvIds[:,0],:]
        sV = regionCenters[uvIds[:,1],:]

        regionCenters.resize((1,1))
        del regionCenters
        gc.collect()

        allFeats.append( (sU - sV)**2 )

        sU = sV.resize((1,1))
        sV = sV.resize((1,1))
        del sU
        del sV
        gc.collect()

        allFeats = np.concatenate(allFeats, axis = 1)

        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Region Features in: " + str(t_feats) + " s")

        self.output().write( np.nan_to_num(allFeats).astype('float32') )


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "RegionFeatures_%s.h5" % (segFile,) )
        return HDF5DataTarget( save_path )


class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()
    keepOnlyXY = luigi.BoolParameter(default = False)
    keepOnlyZ = luigi.BoolParameter(default = False)


    # For now we can't set these any more
    #filterNames = luigi.ListParameter(default = [ "gaussianSmoothing", "hessianOfGaussianEigenvalues", "laplacianOfGaussian"] )
    #sigmas = luigi.ListParameter(default = [1.6, 4.2, 8.3] )

    def requires(self):
        return StackedRegionAdjacencyGraph(self.pathToSeg), InputData(self.pathToInput)

    def run(self):

        inp = self.input()
        rag = inp[0].read()

        assert not(self.keepOnlyXY and self.keepOnlyZ)

        inp[1].open()
        data = inp[1].get()

        t_feats = time.time()

        edge_features = nifty.graph.rag.accumulateEdgeFeaturesFromFilters(rag, data, -1) #, nthreads)
        if self.keepOnlyXY:
            transitionEdge = rag.totalNumberOfInSliceEdges
            edge_features = edge_features[:transitionEdge]
        if self.keepOnlyZ:
            transitionEdge = rag.totalNumberOfInSliceEdges
            edge_features = edge_features[transitionEdge:]

        t_feats = time.time() - t_feats
        workflow_logger.info("Calculated Edge Features in: " + str(t_feats) + " s")

        self.output().write(edge_features)


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        inpFile = os.path.split(self.pathToInput)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "EdgeFeatures_%s_%s" % (segFile,inpFile)  )
        if self.keepOnlyXY:
            save_path += '_xy'
        if self.keepOnlyZ:
            save_path += '_z'
        save_path += '.h5'
        return HDF5DataTarget( save_path )


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
