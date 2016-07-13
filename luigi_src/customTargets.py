from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import logging

import vigra
import cPickle as pickle
import os


class ChunkedTarget(FileSystemTarget):
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
        super(ChunkedTarget, self).__init__(path)

    def open(self, mode = vigra.ReadOnly):

        self.array = vigra.ChunkedArrayHDF5(self.path, mode = mode)

    def write(self, start, data):
        if not os.path.exists(self.path):
            self.makedirs()
        self.array.commitSubarray(start, data)

    def read(self, start, stop):
        return self.array.checkoutSubArray(start, stop)


# vigra impl
class HDF5Target(FileSystemTarget):
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
        super(HDF5Target, self).__init__(path)

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, data, key = 'data', compression = 'gzip'):
        self.makedirs()
        vigra.writeHDF5(data, self.path, key, compression = compression)

    def read(self, key = 'data'):
        return vigra.readHDF5(self.path, key)


class PickleTarget(FileSystemTarget):
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
        super(PickleTarget, self).__init__(path)

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, data):
        self.makedirs()
        with open(self.path, 'w') as f:
            pickle.dump(data, f)

    def read(self):
        with open(self.path, 'r') as f:
            return pickle.load(f)


class RagTarget(FileSystemTarget):
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
        super(RagTarget, self).__init__(path)

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, rag):
        #FIXME
        #assert isinstance(rag, vigra.graphs.regionAdjacencyGraph)
        self.makedirs()
        rag.writeHDF5(self.path, "data")

    def read(self):
        return vigra.graphs.loadGridRagHDF5(self.path, "data")
