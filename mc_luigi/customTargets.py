from __future__ import division, print_function

from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import os
import numpy as np
import pickle

import vigra
import h5py

# import the proper nifty version
try:
    import nifty.graph.rag as nrag
    import nifty.hdf5 as nh5
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
        import nifty_with_cplex.hdf5 as nh5
    except ImportError:
        import nifty_with_gurobi.graph.rag as nrag
        import nifty_with_gurobi.hdf5 as nh5


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
        self.opened = False


    def has_offsets(self, key='data'):
        with h5py.File(self.path) as f:
            ds = f[key]
            if 'offset_front' in ds.attrs.keys():
                assert 'offset_back' in ds.attrs.keys()
                return True
            else:
                return False


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

        # open an existing dataset
        if shape is None:
            assert chunkShape is None

            self.arrays[key] = nh5.hdf5Array(self.dtype, self.h5_file, key)

            # check if any offsets were added to the array
            if self.has_offsets(key):
                print("Loading with offsets")
                with h5py.File(self.path) as f:
                    ds = f[key]
                    offset_front = ds.attrs.get('offset_front')
                    offset_back = ds.attrs.get('offset_back')
                    print(offset_front)
                    print(offset_back)
                    self.arrays[key].setOffsetFront(offset_front)
                    self.arrays[key].setOffsetBack(offset_back)

        # create a new dataset
        else:
            # if we no chunk chape was given, use default chunks
            if chunkShape is not None:
                assert len(chunkShape) == len(shape), str(len(chunkShape)) + " , " + str(len(shape))
                for dd in range(len(shape)):
                    assert chunkShape[dd] <= shape[dd]
            else:
                chunkShape = [1, min(shape[1], 512), min(shape[2], 512)]

            self.arrays[key] = nh5.hdf5Array(
                self.dtype, self.h5_file, key, shape, chunkShape, compression=self.compression
            )

    # add offsets to the nh5 array
    def set_offsets(self, offset_front, offset_back, key='data', serialize_offsets=True):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        self.arrays[key].setOffsetFront(offset_front)
        self.arrays[key].setOffsetBack(offset_back)
        # serialize the offsets
        if serialize_offsets:
            self.serialize_offsets(offset_front, offset_back, key)

    def serialize_offsets(self, offset_front, offset_back, key='data'):
        with h5py.File(self.path) as f:
            ds = f[key]
            ds.attrs.create('offset_front', offset_front)
            ds.attrs.create('offset_back', offset_back)

    def close(self):
        assert self.opened
        assert self.h5_file is not None
        nh5.closeFile(self.h5_file)
        self.opened = False

    def write(self, start, data, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        assert self.opened
        # to avoid errors in python glue code
        start = list(start)
        self.arrays[key].writeSubarray(start, data)

    def read(self, start, stop, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        assert self.opened
        # to avoid errors in python glue code
        start = list(start)
        stop = list(stop)
        return self.arrays[key].readSubarray(start, stop)

    def get(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        assert self.opened
        return self.arrays[key]

    def shape(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        assert self.opened
        return self.arrays[key].shape

    def chunk_shape(self, key='data'):
        if key not in self.arrays:
            raise KeyError("Key does not name a valid dataset in H5 file.")
        assert self.opened
        return self.arrays[key].chunkShape

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
                print("The key", key, "is not in", f.keys())
                raise KeyError("Key not found!")
        return vigra.readHDF5(self.path, key)
