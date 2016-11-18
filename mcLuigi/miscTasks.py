import luigi
import os
import numpy as np

from dataTasks import StackedRegionAdjacencyGraph
from customTargets import HDF5DataTarget
from pipelineParameter import PipelineParameter


# we don't really need this anymore, because we can only just use edgeId > rag.numberOfInnerSliceEdges
class EdgeIndications(luigi.Task):

    pathToSeg = luigi.Parameter()
    keyToSeg = luigi.Parameter(default = "data")

    def requires(self):
        return StackedRegionAdjacencyGraph(self.pathToSeg, self.keyToSeg)

    def run(self):
        rag = self.input().read()
        nEdges = rag.numberOfEdges
        nInner = rag.numberOfInSliceEdges
        nOuter = rag.numberOfInBetweenSliceEdges

        assert nInner + nOuter == nEdges, "Number of inner (%i) + outer (%i) edges must match total number of edges (%i)" % (nInner, nOuter, nEdges)

        edgeIndications = np.ones(nEdges, dtype = np.uint8)

        # inner / xy edges are indicated with a 1, outer / z edges with a 0
        edgeIndications[nInner:] = 0

        self.output().write(edge_indications)

    def output(self):
        return HDF5Target( os.path.join( PipelineParameter().cache, "EdgeIndications.h5" ) )
