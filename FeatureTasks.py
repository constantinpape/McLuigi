# Multicut Pipeline implemented with luigi
# Taksks for Feature Calculation

import luigi
from CustomTargets import HDF5Target
from PipelineParameter import *
from DataTasks import InputData, RegionAdjacencyGraph, ExternalSegmentation

import logging

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


# implement this as function, because we don't want to cache the filters!
def filter_vigra(PathToInput, FilterName, Sigma, Anisotropy):

    inp = vigra.readHDF5(PathToInput, "data")

    # TODO assert thtat this exists
    eval_filter = eval( ".".join( ["vigra", "filters", FilterName] ) )

    # calculate filter purely in 2d
    if Anisotropy > PipelineParameter().max_aniso:
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
        if Anisotropy > 1.:
            sig = (Sigma, Sigma, Sigma / Anisotropy)
        else:
            sig = Sigma

        res = eval_filter( inp, sig )

    return res



class EdgeFeatures(luigi.Task):

    # input over which filters are calculated and features accumulated
    PathToInput = luigi.Parameter()
    # current oversegmentation
    PathToSeg = luigi.Parameter()
    FeatureParameter  = luigi.DictParameter()

    def requires(self):
        return RegionAdjacencyGraph(self.PathToSeg)

    def run(self):
        edge_features = []
        rag = self.input().read()
        for filter_name in self.FeatureParameter["filternames"]:
            for sigma in self.FeatureParameter["sigmas"]:
                filt = filter_vigra(self.PathToInput, filter_name, sigma, self.FeatureParameter["anisotropy"])

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
        # TODO make filter_list and sigmas hashable and add them here
        return HDF5Target( os.path.join( PipelineParameter().cache,
                "_".join([ "EdgeFeatures", os.path.split(self.PathToInput)[1],
                    os.path.split(self.PathToSeg)[1],
                    str(self.FeatureParameter["anisotropy"]) ]) ) + ".h5")
        #return HDF5Target(
        #    os.path.join(
        #        PipelineParameter().cache, os.path.split(self.PathToInput)[1]) + ".h5")
