from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import logging

import os
import numpy as np
import cPickle as pickle

import nifty
import vigra


class HDF5VolumeTarget(FileSystemTarget):
    """
    Target for volumetric data larger than RAM
    """
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

    def __init__(self, path, dtype, key = "data"):
        super(HDF5VolumeTarget, self).__init__(path)
        self.key = key
        self.dtype = dtype
        self.shape = None
        self.chunkShape = None

    def open(self, shape = None, chunkShape = None):
        # open an existing hdf5 file
        if os.path.exists(self.path):
            h5_file = nifty.hdf5.openFile(self.path)
            # set the dtype #TODO (could we do this in a more elegant way?)
            if self.dtype == np.float32:
                self.array = nifty.hdf5.Hdf5ArrayFloat32(h5_file, self.key)
            elif self.dtype == np.float64:
                self.array = nifty.hdf5.Hdf5ArrayFloat64(h5_file, self.key)
            elif self.dtype == np.uint8:
                self.array = nifty.hdf5.Hdf5ArrayUInt8(h5_file,   self.key)
            elif self.dtype == np.uint32:
                self.array = nifty.hdf5.Hdf5ArrayUInt32(h5_file,  self.key)
            elif self.dtype == np.uint64:
                self.array = nifty.hdf5.Hdf5ArrayUInt64(h5_file,  self.key)
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(self.dtype),))
            self.shape = self.array.shape
            self.chunkShape = self.array.chunkShape
        # create a new file
        else:
            self.makedirs()
            h5_file = nifty.hdf5.createFile(self.path)
            # shape and chunk shape
            assert shape != None, "HDF5VolumeTarget needs to be initialised with a shape, when creating a new file"
            self.shape = shape
            if chunkShape != None:
                self.chunkShape = chunkShape
            else:
                self.chunkShape = [1, min(self.shape[1], 512), min(self.shape[2], 512)]

            # set the accordingly #TODO (could we do this in a more elegant way?)
            if self.dtype == np.float32:
                self.array = nifty.hdf5.Hdf5ArrayFloat32(h5_file, self.key, self.shape, self.chunkShape)
            elif self.dtype == np.float64:
                self.array = nifty.hdf5.Hdf5ArrayFloat64(h5_file, self.key, self.shape, self.chunkShape)
            elif self.dtype == np.uint8:
                self.array = nifty.hdf5.Hdf5ArrayUInt8(h5_file,   self.key, self.shape, self.chunkShape)
            elif self.dtype == np.uint32:
                self.array = nifty.hdf5.Hdf5ArrayUInt32(h5_file,  self.key, self.shape, self.chunkShape)
            elif self.dtype == np.uint64:
                self.array = nifty.hdf5.Hdf5ArrayUInt64(h5_file,  self.key, self.shape, self.chunkShape)
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(dtype),))


    def write(self, start, data):
        if not os.path.exists(self.path):
            self.makedirs()
        self.array.writeSubarray(start, data)

    def read(self, start, stop):
        return self.array.readSubarray(start, stop)

    def get(self):
        return self.array


# TODO this is not really necessary and it should be possible to do it all with the nifty target
class HDF5DataTarget(FileSystemTarget):
    """
    Target for in ram data
    """
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
        super(HDF5DataTarget, self).__init__(path)

    def open(self):
        raise AttributeError("Not implemented")

    def write(self, data, key = "data", compression = None):
        if compression != None:
            vigra.writeHDF5(data, self.path, key, compression = compression)
        else:
            vigra.writeHDF5(data, self.path, key)

    def read(self, key = "data" ):
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


# serializing the nifty rag
class StackedRagTarget(FileSystemTarget):
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
        super(StackedRagTarget, self).__init__(path)

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, rag, labelsPath, labelsKey = "data"):
        self.makedirs()
        serialization = rag.serialize()
        vigra.writeHDF5(serialization, self.path, "data")
        vigra.writeHDF5(labelsPath, self.path, "labelsPath")
        vigra.writeHDF5(labelsKey, self.path, "labelsKey")

    def read(self):
        labelsPath = vigra.readHDF5(self.path, "labelsPath")
        labelsKey = vigra.readHDF5(self.path, "labelsKey")
        serialization = vigra.readHDF5(self.path, "data")

        h5_file = nifty.hdf5.openFile(labelsPath)
        labels = nifty.hdf5.Hdf5ArrayUInt32(h5_file, labelsKey)

        nNodes = serialization[0]
        rag = nifty.graph.rag.deserializeGridRagStacked2DHdf5(labels, nNodes, serialization)

        return rag
