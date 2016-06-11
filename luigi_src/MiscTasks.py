import luigi

from DataTasks import RegionAdjacencyGraph


class EdgeIndications(luigi.Task):

    PathToSeg = luigi.Parameter()

    def requires(self):
        return RegionAdjacencyGraph(self.PathToSeg)

    def run(self):
        rag = self.input().read()
        n_edges = rag.edgeNum
        edge_indications = np.zeros(n_edges)
        # TODO get rid of this loop
        for edge_id in range( n_edges ):
            edge_coords = rag.edgeCoordinates(edge_id)
            z_coords = edge_coords[:,2]
            z = np.unique(z_coords)
            assert z.size == 1, "Edge indications can only be calculated for flat superpixel" + str(z)
            # check whether we have a z or a xy edge
            if z - int(z) == 0.:
                # xy-edge!
                edge_indications[edge_id] = 1
            else:
                # z-edge!
                edge_indications[edge_id] = 0
        self.output().write(edge_indications)

    def output(self):
        seg_name = os.path.split(self.PathToSeg)[1][:-3]
        return HDF5Target( os.path.join( PipelineParameter().cache,
            "EdgeIndications_" + seg_name ".h5" ) )
