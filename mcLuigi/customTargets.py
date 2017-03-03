from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import logging

import os
import numpy as np
import cPickle as pickle

import nifty
import vigra
import h5py


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

    def __init__(self, path, dtype, key = "data", compression = -1):
        super(HDF5VolumeTarget, self).__init__(path)
        self.key = key
        self.dtype = dtype
        self.shape = None
        self.chunkShape = None
        self.h5_file = None
        self.compression = compression


    def open(self, shape = None, chunkShape = None):

        # FIXME hacked in dirty for sample D -> expose this somehow!
        # TODO alternatively disable the chunk by setting it to zero as we don't really need it!
        # cache slots: max performance: 100 x number of chunks in cache + prime number, settle to 10 for now
        # chunks in cache: 4E9 / (512**2 * 4) = 3814 -> next_prime_of (38140)
        #hashTableSize = 38149L
        hashTableSize = 7
        # cache size:  size of cache in bytes -> need aboout 16 gig for 40 threads... -> let's go for 4 gigs now
        #nBytes  = 2000000000L
        nBytes  = 4000
        # cache setting: set to 1 because we always do read / write only
        rddc = 1.

        cacheSettings = nifty.hdf5.CacheSettings(hashTableSize,nBytes,rddc)

        # open an existing hdf5 file
        if os.path.exists(self.path):
            self.h5_file = nifty.hdf5.openFile(self.path)#, cacheSettings)
            # set the dtype #TODO (could we do this in a more elegant way?)
            if np.dtype(self.dtype) == np.dtype("float32"):
                self.array = nifty.hdf5.Hdf5ArrayFloat32(self.h5_file, self.key)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("float64"):
                self.array = nifty.hdf5.Hdf5ArrayFloat64(self.h5_file, self.key)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint8"):
                self.array = nifty.hdf5.Hdf5ArrayUInt8(self.h5_file,   self.key)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint32"):
                self.array = nifty.hdf5.Hdf5ArrayUInt32(self.h5_file,  self.key)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint64"):
                self.array = nifty.hdf5.Hdf5ArrayUInt64(self.h5_file,  self.key)#, cache_slots, cache_size, cache_setting)
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(self.dtype),))
            self.shape = self.array.shape
            self.chunkShape = self.array.chunkShape

        # create a new file
        else:
            self.makedirs()
            self.h5_file = nifty.hdf5.createFile(self.path)#, cacheSettings)
            # shape and chunk shape
            assert shape != None, "HDF5VolumeTarget needs to be initialised with a shape, when creating a new file"
            self.shape = shape
            if chunkShape != None:
                assert len(chunkShape) == len(self.shape), str(len(chunkShape)) + " , " + str(len(self.shape))
                for dd in range(len(self.shape)):
                    assert chunkShape[dd] <= self.shape[dd]
                self.chunkShape = chunkShape
            else:
                self.chunkShape = [1, min(self.shape[1], 512), min(self.shape[2], 512)]

            # set the accordingly #TODO (could we do this in a more elegant way?)
            if np.dtype(self.dtype) == np.dtype("float32"):
                self.array = nifty.hdf5.Hdf5ArrayFloat32(self.h5_file, self.key, self.shape, self.chunkShape, compression = self.compression)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("float64"):
                self.array = nifty.hdf5.Hdf5ArrayFloat64(self.h5_file, self.key, self.shape, self.chunkShape, compression = self.compression)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint8"):
                self.array = nifty.hdf5.Hdf5ArrayUInt8(self.h5_file,   self.key, self.shape, self.chunkShape, compression = self.compression)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint32"):
                self.array = nifty.hdf5.Hdf5ArrayUInt32(self.h5_file,  self.key, self.shape, self.chunkShape, compression = self.compression)#, cache_slots, cache_size, cache_setting)
            elif np.dtype(self.dtype) == np.dtype("uint64"):
                self.array = nifty.hdf5.Hdf5ArrayUInt64(self.h5_file,  self.key, self.shape, self.chunkShape, compression = self.compression)#, cache_slots, cache_size, cache_setting)
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(self.dtype),))

    def close(self):
        assert self.h5_file != None
        nifty.hdf5.closeFile(self.h5_file)


    def write(self, start, data):
        # to avoid errors in python glue code
        start = list(map(long,start))
        try:
            self.array.writeSubarray(start, data)
        except AttributeError:
            raise RuntimeError("You must call open once before calling read or write!")

    def read(self, start, stop):
        # to avoid errors in python glue code
        start = list(map(long,start))
        stop = list(map(long,stop))
        try:
            return self.array.readSubarray(start, stop)
        except AttributeError:
            raise RuntimeError("You must call open once before calling read or write!")

    def get(self):
        return self.array


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
        self.makedirs()
        if compression != None:
            vigra.writeHDF5(data, self.path, key, compression = compression)
        else:
            vigra.writeHDF5(data, self.path, key)

    def writeVlen(self, data, key = 'data'):
        self.makedirs()
        with h5py.File(self.path) as f:
            dt = h5py.special_dtype(vlen=np.dtype(data[0].dtype))
            ds = f.create_dataset(key, data = data, dtype = dt)

    def read(self, key = "data" ):
        return vigra.readHDF5(self.path, key)

    def shape(self, key = "data"):
        with h5py.File(self.path) as f:
            shape = f[key].shape
        return shape


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
        nifty.graph.rag.writeStackedRagToHdf5(rag, self.path)
        vigra.writeHDF5(labelsPath, self.path, "labelsPath")
        vigra.writeHDF5(labelsKey, self.path, "labelsKey")

    # read and deserialize the rag
    def read(self):
        labelsPath = vigra.readHDF5(self.path, "labelsPath")
        labelsKey = vigra.readHDF5(self.path, "labelsKey")

        h5_file = nifty.hdf5.openFile(labelsPath)
        labels = nifty.hdf5.Hdf5ArrayUInt32(h5_file, labelsKey)
        nNodes = vigra.readHDF5(self.path, "numberOfNodes")

        return nifty.graph.rag.readStackedRagFromHdf5(labels, nNodes, self.path)

    # only read sub-parts
    def readKey(self, key):
        with h5py.File(self.path,'r') as f:
            if not key in f.keys():
                print "The key", key, "is not in", f.keys()
                raise KeyError("Key not found!")
        return vigra.readHDF5(self.path, key)

    def shape(self):
        return rag.shape
