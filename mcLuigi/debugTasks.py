# vigra implementations for debugging rag and features

import luigi
from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

from pipelineParameter import PipelineParameter
from dataTasks import StackedRegionAdjacencyGraph, ExternalSegmentation, InputData
from customTargets import HDF5DataTarget

from tools import config_logger

import os
import logging
import json
import time

import numpy as np
import vigra
import nifty

from concurrent import futures

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


def get_local_vigra_features(learnXYOnly = False, learnZOnly = False):

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
        feature_tasks.append( EdgeFeaturesVigra(input_data[0], inputs["seg"]) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[0])

    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeFeaturesVigra(input_data[1], inputs["seg"] ) ) #, filternames, sigmas) )
        workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[1])

    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        # TODO should try it on probmaps. For big data we might spare shipping the raw data!
        feature_tasks.append( RegionFeaturesVigra(input_data[0], inputs["seg"]) )
        workflow_logger.debug("Calculating Region Features")

    return feature_tasks


# serializing the nifty rag
class VigraRagTarget(FileSystemTarget):
    fs = LocalFileSystem()

    def makedirs(self):
        """
        Create all parent folders if they do not exist.
        """
        normpath = os.path.normpath(self.path)
        parentfolder = os.path.dirname(normpath)
        if parentfolder:
            try:
                os.makedirs(parentfolder)
            except OSError:
                pass

    def __init__(self, path):
        super(VigraRagTarget, self).__init__(path)

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, rag):
        self.makedirs()
        rag.writeHDF5(self.path, 'data')

    def read(self):
        return vigra.graphs.loadGridRagHDF5(self.path, 'data')


class VigraRag(luigi.Task):
    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default = "data")

    # not really necessary right now, but maybe the rag syntax will change
    def requires(self):
        return ExternalSegmentation(self.pathToSeg)

    def run(self):

        seg = self.input()
        seg.open()
        seg = seg.read([0L,0L,0L],seg.shape)
        seg = seg.T

        rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg.shape), seg)
        self.output().write(rag)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = "VigraRag_%s.h5" % (segFile,)
        return VigraRagTarget( os.path.join(PipelineParameter().cache, save_path) )


class NiftyToVigraEdges(luigi.Task):
    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default = "data")

    def requires(self):
        return {"vigraRag" : VigraRag(self.pathToSeg), "niftyRag" : StackedRegionAdjacencyGraph(self.pathToSeg)}

    def run(self):

        inp = self.input()
        vrag = inp["vigraRag"].read()
        nrag = inp["niftyRag"].read()

        nNodes = nrag.numberOfNodes
        assert nNodes == vrag.nodeNum, str(nNodes) + " , " + str(vrag.nodeNum)

        nEdges = nrag.numberOfEdges
        assert nEdges == vrag.edgeNum, str(nEdges) + " , " + str(vrag.edgeNum)

        edgeDict = np.zeros(nEdges, dtype = 'uint32' )
        uvIds = np.sort(vrag.uvIds(), axis = 1)

        for v_id in xrange(nEdges):
            n_id = int(nrag.findEdge(uvIds[v_id,0],uvIds[v_id,1]))
            assert n_id >= 0, " , ".join([str(n_id),str(v_id),str(uvIds[v_id,0]),str(uvIds[v_id,1])])
            assert n_id < nEdges
            edgeDict[n_id] = v_id

        assert (uvIds[edgeDict] == nrag.uvIds()).all()

        self.output().write(edgeDict)

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = "NiftyToVigraEdges_%s.h5" % (segFile,)
        return HDF5DataTarget( os.path.join(PipelineParameter().cache, save_path) )




class EdgeFeaturesVigra(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()

    def requires(self):
        return { "rag" : VigraRag(self.pathToSeg), "data" : InputData(self.pathToInput), "n2vEdges" : NiftyToVigraEdges(self.pathToSeg) }

    def run(self):

        inp = self.input()
        rag = inp["rag"].read()


        def _accumulate_filter_over_edge(filt):
            feats_return = []
            if len(filt.shape) == 3:
                gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, filt)
                edgeFeats     = rag.accumulateEdgeStatistics(gridGraphEdgeIndicator)
                feats_return.append(edgeFeats)
            elif len(filt.shape) == 4:
                for c in range(filt.shape[3]):
                    gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(
                            rag.baseGraph, filt[:,:,:,c] )
                    edgeFeats     = rag.accumulateEdgeStatistics(gridGraphEdgeIndicator)
                    feats_return.append(edgeFeats)
            return feats_return


        data = inp["data"]
        data.open()
        data = data.read([0L,0L,0L],data.shape)

        data = data.T

        import fastfilters
        filter_names = [ "fastfilters.gaussianSmoothing",
                         "fastfilters.laplacianOfGaussian",
                         "fastfilters.hessianOfGaussianEigenvalues"]
        sigmas = [1.6, 4.2, 8.2]

        edge_features = []

        for fname in filter_names:
            filter_fu = eval(fname)
            for sigma in sigmas:

                with futures.ThreadPoolExecutor(max_workers = 40 ) as executor:
                    tasks = []
                    for z in xrange(data.shape[0]):
                        # read input for current slice
                        tasks.append( executor.submit(filter_fu, data[z,:,:] , sigma ) )
                    response = [task.result() for task in tasks]

                if response[0].ndim == 2:
                    response = np.concatenate([re[:,:,None] for re in response], axis = 2)
                elif response[0].ndim == 3:
                    response = np.concatenate([re[:,:,None,:] for re in response], axis = 2)

                # accumulate over the edge
                feats = np.concatenate(_accumulate_filter_over_edge(response), axis = 1)
                edge_features.append(feats)

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == rag.edgeNum, str(edge_features.shape[0]) + " , " +str(rag.edgeNum)
        # remove NaNs
        edge_features = np.nan_to_num(edge_features)
        n2vEdges = inp["n2vEdges"].read()
        edge_features = edge_features[n2vEdges]
        self.output().write(edge_features)


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        inpFile = os.path.split(self.pathToInput)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "EdgeFeaturesVigra_%s_%s.h5" % (segFile,inpFile)  )
        return HDF5DataTarget( save_path )


class RegionFeaturesVigra(luigi.Task):

    # input over which filters are calculated and features accumulated
    pathToInput = luigi.Parameter()
    # current oversegmentation
    pathToSeg = luigi.Parameter()

    def requires(self):
        return { "rag" : VigraRag(self.pathToSeg), "data" : InputData(self.pathToInput),
                "seg" : ExternalSegmentation(self.pathToSeg), "n2vEdges" : NiftyToVigraEdges(self.pathToSeg) }

    def run(self):

        inp = self.input()
        rag = inp["rag"].read()

        data = inp["data"]
        data.open()
        data = data.read([0L,0L,0L],data.shape)
        data = data.T

        seg = inp["seg"]
        seg.open()
        seg = seg.read([0L,0L,0L],seg.shape)
        seg = seg.T

        # list of the region statistics, that we want to extract
        statistics =  [ "Count", "Kurtosis", #"Histogram",
                        "Maximum", "Minimum", "Quantiles",
                        "RegionRadii", "Skewness", "Sum",
                        "Variance", "Weighted<RegionCenter>", "RegionCenter"]

        region_statistics = vigra.analysis.extractRegionFeatures(
                data.astype(np.float32), seg,
                features = statistics )

        import gc

        regStats = []

        for regStatName in statistics[:9]:
            regStat = region_statistics[regStatName]
            if regStat.ndim == 1:
                regStats.append(regStat[:,None])
            else:
                regStats.append(regStat)
        regStats = np.concatenate(regStats, axis=1)

        regCenters = []
        for regStatName in  statistics[9:]:
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

        uv_ids = rag.uvIds()

        fU = regStats[uv_ids[:,0],:]
        fV = regStats[uv_ids[:,1],:]

        allFeat = [
                np.minimum(fU, fV),
                np.maximum(fU, fV),
                np.abs(fU - fV),
                fU + fV
            ]

        fV = fV.resize((1,1))
        fU = fU.resize((1,1))
        del fU
        del fV
        gc.collect()

        sU = regCenters[uv_ids[:,0],:]
        sV = regCenters[uv_ids[:,1],:]
        allFeat.append( (sU - sV)**2 )

        sV = sV.resize((1,1))
        sU = sU.resize((1,1))
        del sU
        del sV
        gc.collect()

        allFeat = np.concatenate(allFeat, axis = 1)

        n2vEdges = inp["n2vEdges"].read()
        allFeat = allFeat[n2vEdges]

        self.output().write( np.nan_to_num(allFeat) )


    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "EdgeFeaturesVigra_%s.h5" % (segFile,)  )
        return HDF5DataTarget( save_path )


class McPromblemVigraFromFile(luigi.Task):

    pathToSeg = luigi.Parameter()
    pathToVigraWeights = luigi.Parameter(
        default = '/home/constantin/Work/home_hdd/cache/cremi/sample_A_val_small/probs_to_energies_0_-2597982656216043428.h5')

    def requires(self):
        return NiftyToVigraEdges(self.pathToSeg), StackedRegionAdjacencyGraph(self.pathToSeg)

    def run(self):
        n2vEdges = self.input()[0].read()
        weights = vigra.readHDF5(self.pathToVigraWeights, 'data')
        weights = weights[n2vEdges]

        rag = self.input()[1].read()
        uvIds = rag.uvIds()
        nVariables = uvIds.max() + 1

        g = nifty.graph.UndirectedGraph(int(nVariables))
        g.insertEdges(uvIds)

        out = self.output()

        assert g.numberOfEdges == weights.shape[0]
        out.write( g.serialize(), "graph" )
        out.write( weights, "costs")

    def output(self):
        segFile = os.path.split(self.pathToSeg)[1][:-3]
        save_path = os.path.join( PipelineParameter().cache, "McProblemVigraFromFile_%s.h5" % (segFile,)  )
        return HDF5DataTarget( save_path )
