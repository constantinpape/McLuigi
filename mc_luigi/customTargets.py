from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import os
import numpy as np
import cPickle as pickle

import vigra
import h5py

# import the proper nifty version
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty
    except ImportError:
        import nifty_with_gurobi as nifty
import nifty.graph.rag as nrag
import nifty.hdf5 as nh5


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

    def __init__(self, path, dtype, defaultKey='data', compression=-1):
        super(HDF5VolumeTarget, self).__init__(path)
        self.h5_file = None
        self.dtype = dtype
        self.compression = compression
        self.arrays = {}
        self.shapes = {}
        self.chunk_shapes = {}
        self.opened = False

    def openExisting(self):
        with h5py.File(self.path) as f:
            keys = f.keys()
        for key in keys:
            self.open(key=key)

    def open(self, shape=None, chunkShape=None, key='data'):

        # check if this file is already open
        if not self.opened:
            # open an existing hdf5 file
            if os.path.exists(self.path):
                self.h5_file = nh5.openFile(self.path)
            # create a new file
            else:
                self.makedirs()
                self.h5_file = nh5.createFile(self.path)  # , cacheSettings)
            self.opened = True

        newDataset = True
        with h5py.File(self.path) as f:
            if key in f.keys():
                newDataset = False

        # open an existing dataset
        if not newDataset:
            assert shape is None
            assert chunkShape is None

            # set the dtype #TODO (could we do this in a more elegant way?)
            if np.dtype(self.dtype) == np.dtype("float32"):
                self.arrays[key] = nh5.Hdf5ArrayFloat32(self.h5_file, key)
            elif np.dtype(self.dtype) == np.dtype("float64"):
                self.arrays[key] = nh5.Hdf5ArrayFloat64(self.h5_file, key)
            elif np.dtype(self.dtype) == np.dtype("uint8"):
                self.arrays[key] = nh5.Hdf5ArrayUInt8(self.h5_file, key)
            elif np.dtype(self.dtype) == np.dtype("uint32"):
                self.arrays[key] = nh5.Hdf5ArrayUInt32(self.h5_file, key)
            elif np.dtype(self.dtype) == np.dtype("uint64"):
                self.arrays[key] = nh5.Hdf5ArrayUInt64(self.h5_file, key)
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(self.dtype),))

        # create a new dataset
        else:
            # shape and chunk shape
            assert shape is not None, "HDF5VolumeTarget needs to be initialised with a shape, when creating a new file"
            if chunkShape is not None:
                assert len(chunkShape) == len(shape), str(len(chunkShape)) + " , " + str(len(shape))
                for dd in range(len(shape)):
                    assert chunkShape[dd] <= shape[dd]
            else:
                chunkShape = [1, min(shape[1], 512), min(shape[2], 512)]

            if np.dtype(self.dtype) == np.dtype("float32"):
                self.arrays[key] = nh5.Hdf5ArrayFloat32(
                    self.h5_file, key, shape, chunkShape, compression=self.compression
                )
            elif np.dtype(self.dtype) == np.dtype("float64"):
                self.arrays[key] = nh5.Hdf5ArrayFloat64(
                    self.h5_file, key, shape, chunkShape, compression=self.compression
                )
            elif np.dtype(self.dtype) == np.dtype("uint8"):
                self.arrays[key] = nh5.Hdf5ArrayUInt8(
                    self.h5_file, key, shape, chunkShape, compression=self.compression
                )
            elif np.dtype(self.dtype) == np.dtype("uint32"):
                self.arrays[key] = nh5.Hdf5ArrayUInt32(
                    self.h5_file, key, shape, chunkShape, compression=self.compression
                )
            elif np.dtype(self.dtype) == np.dtype("uint64"):
                self.arrays[key] = nh5.Hdf5ArrayUInt64(
                    self.h5_file, key, shape, chunkShape, compression=self.compression
                )
            else:
                raise RuntimeError("Datatype %s not supported!" % (str(self.dtype),))
        self.shapes[key] = self.arrays[key].shape
        self.chunk_shapes[key] = self.arrays[key].chunkShape

    def close(self):
        assert self.opened
        assert self.h5_file is not None
        nh5.closeFile(self.h5_file)
        self.opened = False

    def write(self, start, data, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        # to avoid errors in python glue code
        start = list(map(long, start))
        try:
            self.arrays[key].writeSubarray(start, data)
        except AttributeError:
            raise RuntimeError("You must call open once before calling read or write!")

    def read(self, start, stop, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        # to avoid errors in python glue code
        start = list(map(long, start))
        stop = list(map(long, stop))
        try:
            return self.arrays[key].readSubarray(start, stop)
        except AttributeError:
            raise RuntimeError("You must call open once before calling read or write!")

    # TODO not exposed in nifty right now
    # def writeLocked(self, start, data, key = 'data'):
    #    if not key in self.arrays:
    #        raise KeyError("Key does not name a valid dataset in H5 file.")
    #    # to avoid errors in python glue code
    #    start = list(map(long,start))
    #    try:
    #        self.arrays[key].writeSubarrayLocked(start, data)
    #    except AttributeError:
    #        raise RuntimeError("You must call open once before calling read or write!")

    # def readLocked(self, start, stop, key = 'data'):
    #    if not key in self.arrays:
    #        raise KeyError("Key does not name a valid dataset in H5 file.")
    #    # to avoid errors in python glue code
    #    start = list(map(long,start))
    #    stop = list(map(long,stop))
    #    try:
    #        return self.arrays[key].readSubarrayLocked(start, stop)
    #    except AttributeError:
    #        raise RuntimeError("You must call open once before calling read or write!")

    def get(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        return self.arrays[key]

    def shape(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        return self.shapes[key]

    def chunk_shape(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        return self.chunk_shapes[key]


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

    def write(self, data, key="data", compression=None):
        self.makedirs()
        if compression is not None:
            vigra.writeHDF5(data, self.path, key, compression=compression)
        else:
            vigra.writeHDF5(data, self.path, key)

    def writeVlen(self, data, key='data'):
        self.makedirs()
        with h5py.File(self.path) as f:
            dt = h5py.special_dtype(vlen=np.dtype(data[0].dtype))
            f.create_dataset(key, data=data, dtype=dt)

    def read(self, key="data"):
        return vigra.readHDF5(self.path, key)

    def shape(self, key="data"):
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


# Folder target that does basically nothing
# wee need this for the sklearn random forest,
# that is pickled to different files in a common folder
class FolderTarget(FileSystemTarget):
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
        super(FolderTarget, self).__init__(path)
        self.path = path

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, data):
        raise AttributeError("Not implemented")

    def read(self):
        raise AttributeError("Not implemented")


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

    def write(self, rag, labelsPath, labelsKey="data"):
        self.makedirs()
        nrag.writeStackedRagToHdf5(rag, self.path)
        vigra.writeHDF5(labelsPath, self.path, "labelsPath")
        vigra.writeHDF5(labelsKey, self.path, "labelsKey")

    # read and deserialize the rag
    def read(self):
        labelsPath = vigra.readHDF5(self.path, "labelsPath")
        labelsKey = vigra.readHDF5(self.path, "labelsKey")

        h5_file = nh5.openFile(labelsPath)
        labels = nh5.Hdf5ArrayUInt32(h5_file, labelsKey)
        nNodes = vigra.readHDF5(self.path, "numberOfNodes")

        return nrag.readStackedRagFromHdf5(labels, nNodes, self.path)

    # only read sub-parts
    def readKey(self, key):
        with h5py.File(self.path, 'r') as f:
            if key not in f.keys():
                print "The key", key, "is not in", f.keys()
                raise KeyError("Key not found!")
        return vigra.readHDF5(self.path, key)
