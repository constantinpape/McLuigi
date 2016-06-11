# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi
from CustomTargets import HDF5Target
from PipelineParameter import *
from DataTasks import InputData, RegionAdjacencyGraph, ExternalSegmentation

import logging
import json

import os
import numpy as np
import vigra


# TODO
# class FilterSven
# class RegionFeatures
# class TopologyFeatures
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
def getLocalFeatures():
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


# TODO class RegionFeatures(luigi.Task)
# TODO class ToplogyFeatures(luigi.Task)

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
        return HDF5Target( os.path.join( PipelineParameter().cache,
            "EdgeFeatures_" + inp_name + ".h5" ) )
