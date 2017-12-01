from __future__ import division, print_function

from luigi.target import FileSystemTarget

import os
import numpy as np
import pickle

import vigra
import h5py
import z5py

from .pipelineParameter import PipelineParameter

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


class BaseTarget(FileSystemTarget):
    """
    Custom target base class
    """
    def __init__(self, path):
        super(BaseTarget, self).__init__(path)

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


# TODO enable zarr format ?!
class N5Target(BaseTarget):
    """
    Target for data in n5 format
    """
    def __init__(self, path):
        super(N5Target, self).__init__(path)
        self.datasets = {}
        self.n5_file = z5py.File(self.path, use_zarr_format=False)

    def __contains__(self, key):
        return key in self.n5_file

    # TODO change compression to blosc as soon as n5 supports it !
    # TODO offset handling, need to implement loading with offsets and offsets in z5
    def open(self, key='data', dtype=None, shape=None, chunks=None, compression='gzip', **compression_opts):
        # if we have already opened the dataset, we don't need to do anything
        if key in self.datasets:
            return self
        # otherwise we need to check, if this dataset exists on file and either open or create it
        if key in self.n5_file:
            self.datasets[key] = self.n5_file[key]
        else:
            # if we need to create the dataset, we need to make sure that
            # dtype, shape and chunks are actually specified
            assert dtype is not None, "Can't open a new dataset if dtype is not specified"
            assert shape is not None, "Can't open a new dataset if shape is not specified"
            assert chunks is not None, "Can't open a new dataset if chunks are not specified"
            self.datasets[key] = self.n5_file.create_dataset(key,
                                                             dtype=dtype,
                                                             shape=shape,
                                                             chunks=chunks,
                                                             compressor=compression,
                                                             **compression_opts)
        # check if any offsets were added to the array
        if self.has_offsets(key):
            ds = self.datasets[key]
            offset_front = ds.attrs.get('offset_front')
            offset_back = ds.attrs.get('offset_back')
            self.set_offsets(offset_front, offset_back, 'data', serialize_offsets=False)
        return self

    def write(self, start, data, key='data'):
        assert key in self.datasets, "Can't write to a dataset that has not been opened"
        self.datasets[key].write_subarray(start, data)

    def read(self, start, stop, key='data'):
        assert key in self.datasets, "Can't read from a dataset that has not been opened"
        return self.datasets[key].read_subarray(start, stop)

    # get the dataset implementation to pass to c++ code
    def get(self, key='data'):
        assert key in self.datasets, "Can't get ds impl for a dataset that has not been opened"
        return self.datasets[key]._impl

    def shape(self, key='data'):
        assert key in self.datasets, "Can't get shape for a dataset that has not been opened"
        return self.datasets[key].shape

    def chunks(self, key='data'):
        assert key in self.datasets, "Can't get chunks for a dataset that has not been opened"
        return self.datasets[key].chunks

    # dummy implementation to be consisteny with HDF5Target
    def close(self):
        pass

    # add offsets to the nh5 array
    def set_offsets(self, offset_front, offset_back, key='data', serialize_offsets=True):
        assert False, "Offsets not implemented in z5py yet"
        assert key in self.datasets, "Can't set offsets for a dataset that has not been opened"
        # TODO implement in z5
        self.datasets[key].set_offset_front(offset_front)
        self.datasets[key].set_offset_back(offset_back)
        # serialize the offsets
        if serialize_offsets:
            self.serialize_offsets(offset_front, offset_back, key)

    def serialize_offsets(self, offset_front, offset_back, key='data'):
        assert key in self.datasets, "Can't serialize offsets for a dataset that has not been opened"
        self.datasets[key].attrs['offset_front'] = offset_front
        self.datasets[key].attrs['offset_back'] = offset_back

    @staticmethod
    def has_offsets(path, key='data'):
        f = z5py.File(path)
        ds = f[key]
        if 'offset_front' in ds.attrs:
            assert 'offset_back' in ds.attrs
            return True
        else:
            return False


class HDF5Target(BaseTarget):
    """
    Target for h5 data larger than RAM
    """
    def __init__(self, path):
        super(HDF5Target, self).__init__(path)
        self.datasets = {}
        self.h5_file = nh5.openFile(self.path) if os.path.exists(self.path) else \
            nh5.createFile(self.path)

    def __contains__(self, key):
        with h5py.File(self.path) as f:
            return key in f

    def open(self, key='data', dtype=None, shape=None, chunks=None, compression='gzip', **compression_opts):

        # if we have already opened the dataset, we don't need to do anything
        if key in self.datasets:
            return self
        # otherwise we need to check, if this dataset exists on file and either open or create it
        with h5py.File(self.path) as fh5:
            has_key = key in fh5
        if has_key:
            self.datasets[key] = nh5.hdf5Array(self.dtype, self.h5_file, key)
        else:
            # if we need to create the dataset, we need to make sure that
            # dtype, shape and chunks are actually specified
            assert dtype is not None, "Can't open a new dataset if dtype is not specified"
            assert shape is not None, "Can't open a new dataset if shape is not specified"
            assert chunks is not None, "Can't open a new dataset if chunks are not specified"
            self.datasets[key] = nh5.hdf5Array(dtype, self.h5_file, key,
                                               shape, chunks,
                                               compression=compression,
                                               **compression_opts)
        # check if any offsets were added to the array
        if self.has_offsets(key):
            ds = self.datasets[key]
            offset_front = ds.attrs.get('offset_front')
            offset_back = ds.attrs.get('offset_back')
            self.set_offsets(offset_front, offset_back, 'data', serialize_offsets=False)
        return self

    def close(self):
        nh5.closeFile(self.h5_file)

    def write(self, start, data, key='data'):
        assert key in self.datasets, "Can't write to a dataset that has not been opened"
        self.datasets[key].writeSubarray(list(start), data)

    def read(self, start, stop, key='data'):
        assert key in self.datasets, "Can't read from a dataset that has not been opened"
        return self.datasets[key].readSubarray(list(start), list(stop))

    def get(self, key='data'):
        assert key in self.datasets, "Can't get ds impl for a dataset that has not been opened"
        return self.datasets[key]

    def shape(self, key='data'):
        assert key in self.datasets, "Can't get shape for a dataset that has not been opened"
        return self.datasets[key].shape

    def chunks(self, key='data'):
        assert key in self.datasets, "Can't get chunks for a dataset that has not been opened"
        return self.datasets[key].chunkShape

    # add offsets to the nh5 array
    def set_offsets(self, offset_front, offset_back, key='data', serialize_offsets=True):
        assert key in self.datasets, "Can't set offsets for a dataset that has not been opened"
        self.datasets[key].setOffsetFront(offset_front)
        self.datasets[key].setOffsetBack(offset_back)
        # serialize the offsets
        if serialize_offsets:
            self.serialize_offsets(offset_front, offset_back, key)

    def serialize_offsets(self, offset_front, offset_back, key='data'):
        assert key in self.datasets, "Can't serialize offsets for a dataset that has not been opened"
        self.datasets[key].attrs.create('offset_front', offset_front)
        self.datasets[key].attrs.create('offset_back', offset_back)

    @staticmethod
    def has_offsets(path, key='data'):
        with h5py.File(path) as f:
            ds = f[key]
            if 'offset_front' in ds.attrs:
                assert 'offset_back' in ds.attrs
                return True
            else:
                return False


# choose n5 / h5 backedn globally
VolumeTarget = N5Target if PipelineParameter().useN5Backend else HDF5Target


class HDF5DataTarget(BaseTarget):
    """
    Target for h5 data in RAM
    """
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


class PickleTarget(BaseTarget):
    """
    Target for pickle data
    """
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
class FolderTarget(BaseTarget):
    """
    Target for multiple files in folder
    """
    def __init__(self, path):
        super(FolderTarget, self).__init__(path)
        self.path = path

    def open(self, mode='r'):
        raise AttributeError("Not implemented")

    def write(self, data):
        raise AttributeError("Not implemented")

    def read(self):
        raise AttributeError("Not implemented")


# TODO enable n5 supported rag
# serializing the nifty rag
class StackedRagTarget(BaseTarget):
    """
    Target for nifty stacked rag
    """
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
