from __future__ import division, print_function

from luigi.target import FileSystemTarget
from luigi.file import LocalFileSystem

import os
import numpy as np
import pickle

import vigra
import h5py
import z5py

from .pipelineParameter import PipelineParameter

# import the proper nifty version
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        import nifty_with_gurobi as nifty
        import nifty_with_gurobi.graph.rag as nrag


# FIXME this should work ...
#if nifty.Configuration.WITH_HDF5:
#    import nifty.hdf5 as nh5
#
#if nifty.Configuration.WITH_Z5:
#    import nifty.z5 as nz5
import nifty.z5 as nz5


class BaseTarget(FileSystemTarget):
    """
    Custom target base class
    """
    fs = LocalFileSystem()

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


class VolumeTarget(BaseTarget):
    """
    Volume target, can hold n5 or hdf5 backend.
    """
    n5_ending = '.n5'
    h5_ending = '.h5'

    def __init__(self, path):
        # FIXME this does nor work as default argument
        use_n5 = PipelineParameter().useN5Backend
        super(BaseTarget, self).__init__(path)
        self.makedirs()
        self._impl = N5Target(self.path) if use_n5 else HDF5Target(self.path)

    @staticmethod
    def file_ending():
        return VolumeTarget.n5_ending if PipelineParameter().useN5Backend else VolumeTarget.h5_ending

    def __contains__(self, key):
        return key in self._impl

    def open(self, key='data', dtype=None, shape=None, chunks=None, **compression_opts):
        return self._impl.open(key=key, dtype=dtype, shape=shape, chunks=chunks,
                               **compression_opts)

    def write(self, start, data, key='data'):
        self._impl.write(start, data, key)

    def read(self, start, stop, key='data'):
        return self._impl.read(start, stop, key)

    def get(self, key='data'):
        return self._impl.get(key)

    def shape(self, key='data'):
        return self._impl.shape(key)

    def chunks(self, key='data'):
        return self._impl.chunks(key)

    def dtype(self, key='data'):
        return self._impl.dtype(key)

    def close(self):
        self._impl.close()

    def keys_on_filesystem(self):
        return self._impl.keys_on_filesystem()

    def keys(self):
        return self._impl.keys()

    # TODO interface for the offsets once implemented in z5


# TODO enable zarr format ?!
class N5Target(object):
    """
    Target for data in n5 format
    """
    def __init__(self, path):
        self.path = path
        self.datasets = {}
        self.n5_file = None

    def _open_file(self):
        self.n5_file = z5py.File(self.path, use_zarr_format=False)

    def __contains__(self, key):
        if self.n5_file is None:
            self._open_file()
        return key in self.n5_file

    def keys_on_filesystem(self):
        # open the n5 file if it wasn't opened yet
        if self.n5_file is None:
            self._open_file()
        return self.n5_file.keys()

    # TODO change compression to blosc as soon as n5 supports it !
    # TODO offset handling, need to implement loading with offsets and offsets in z5
    def open(self, key='data', dtype=None, shape=None, chunks=None, compression='gzip', **compression_opts):

        # open the n5 file if it wasn't opened yet
        if self.n5_file is None:
            self._open_file()

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
        # TODO implement offsets in n5
        # check if any offsets were added to the array
        # if self.has_offsets(key):
        #     ds = self.datasets[key]
        #     offset_front = ds.attrs.get('offset_front')
        #     offset_back = ds.attrs.get('offset_back')
        #     self.set_offsets(offset_front, offset_back, 'data', serialize_offsets=False)

        return self

    def write(self, start, data, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't write to a dataset that has not been opened"
        self.datasets[key].write_subarray(start, data)

    def read(self, start, stop, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't read from a dataset that has not been opened"
        return self.datasets[key].read_subarray(start, stop)

    # get the dataset implementation to pass to c++ code
    def get(self, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't get ds impl for a dataset that has not been opened"
        dtype = self.datasets[key].dtype
        return nz5.datasetWrapper(dtype, os.path.join(self.path, key))

    def shape(self, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't get shape for a dataset that has not been opened"
        return self.datasets[key].shape

    def chunks(self, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't get chunks for a dataset that has not been opened"
        return self.datasets[key].chunks

    def dtype(self, key='data'):
        assert self.n5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't get chunks for a dataset that has not been opened"
        return self.datasets[key].dtype

    def keys(self):
        assert self.n5_file is not None, "Need to open the n5 file first"
        return self.datasets.keys()

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
        assert False, "Offsets not implemented in z5py yet"
        assert key in self.datasets, "Can't serialize offsets for a dataset that has not been opened"
        self.datasets[key].attrs['offset_front'] = offset_front
        self.datasets[key].attrs['offset_back'] = offset_back

    @staticmethod
    def has_offsets(path, key='data'):
        assert False, "Offsets not implemented in z5py yet"
        f = z5py.File(path)
        ds = f[key]
        if 'offset_front' in ds.attrs:
            assert 'offset_back' in ds.attrs
            return True
        else:
            return False


class HDF5Target(object):
    """
    Target for h5 data larger than RAM
    """
    def __init__(self, path):
        self.path = path
        self.datasets = {}
        self.h5_file = None

    def _open_file(self):
        self.h5_file = nh5.openFile(self.path) if os.path.exists(self.path) else \
            nh5.createFile(self.path)

    def __contains__(self, key):
        with h5py.File(self.path) as f:
            return key in f

    def keys_on_filesystem(self):
        with h5py.File(self.path) as f:
            return f.keys()

    def open(self, key='data', dtype=None, shape=None, chunks=None, compression='gzip', **compression_opts):

        # open the h5 file if it is not exisiting already
        if self.h5_file is None:
            self._open_file()

        # if we have already opened the dataset, we don't need to do anything
        if key in self.datasets:
            return self
        # otherwise we need to check, if this dataset exists on file and either open or create it
        with h5py.File(self.path) as fh5:
            has_key = key in fh5
        if has_key:
            if dtype is None:
                with h5py.File(self.path) as f:
                    dtype = f[key].dtype
            self.datasets[key] = nh5.hdf5Array(dtype, self.h5_file, key)
        else:
            # if we need to create the dataset, we need to make sure that
            # dtype, shape and chunks are actually specified
            assert dtype is not None, "Can't open a new dataset if dtype is not specified"
            assert shape is not None, "Can't open a new dataset if shape is not specified"
            assert chunks is not None, "Can't open a new dataset if chunks are not specified"
            clevel = compression_opts.get('level', 4)
            compression_ = -1 if compression != 'gzip' else clevel
            self.datasets[key] = nh5.hdf5Array(dtype, self.h5_file, key,
                                               shape, chunks,
                                               compression=compression_)
        # TODO re-enable support for ofsets once we have this in z5
        # check if any offsets were added to the array
        # if self.has_offsets(key):
        #     ds = self.datasets[key]
        #     offset_front = ds.attrs.get('offset_front')
        #     offset_back = ds.attrs.get('offset_back')
        #     self.set_offsets(offset_front, offset_back, 'data', serialize_offsets=False)
        return self

    def close(self):
        assert self.h5_file is not None, "Need to open the h5 file first"
        nh5.closeFile(self.h5_file)

    def write(self, start, data, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        assert key in self.datasets, "Can't write to a dataset that has not been opened"
        self.datasets[key].writeSubarray(list(start), data)

    def read(self, start, stop, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        assert key in self.datasets, "Can't read from a dataset that has not been opened"
        return self.datasets[key].readSubarray(list(start), list(stop))

    def get(self, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        assert key in self.datasets, "Can't get ds impl for a dataset that has not been opened"
        return self.datasets[key]

    def shape(self, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        assert key in self.datasets, "Can't get shape for a dataset that has not been opened"
        return self.datasets[key].shape

    def chunks(self, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        assert key in self.datasets, "Can't get chunks for a dataset that has not been opened"
        return self.datasets[key].chunkShape

    def dtype(self, key='data'):
        assert self.h5_file is not None, "Need to open the h5 file first"
        with h5py.File(self.path) as f:
            return f[key].dtype

    def keys(self):
        assert self.h5_file is not None, "Need to open the h5 file first"
        return self.datasets.keys()

    # add offsets to the nh5 array
    def set_offsets(self, offset_front, offset_back, key='data', serialize_offsets=True):
        assert self.h5_file is not None, "Need to open the n5 file first"
        assert key in self.datasets, "Can't set offsets for a dataset that has not been opened"
        self.datasets[key].setOffsetFront(offset_front)
        self.datasets[key].setOffsetBack(offset_back)
        # serialize the offsets
        if serialize_offsets:
            self.serialize_offsets(offset_front, offset_back, key)

    def serialize_offsets(self, offset_front, offset_back, key='data'):
        assert self.h5_file is not None, "Need to open the n5 file first"
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


class HDF5DataTarget(BaseTarget):
    """
    Target for h5 data in RAM
    """
    def __init__(self, path):
        super(HDF5DataTarget, self).__init__(path)

    def open(self, shape, dtype, compression=None, chunks=None, key='data'):
        with h5py.File(self.path) as f:
            f.create_dataset(key,
                             shape=shape,
                             compression=compression,
                             chunks=chunks,
                             dtype=dtype)

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

    def writeSubarray(self, start, data, key="data"):
        bb = tuple(slice(sta, sta + sha) for sta, sha in zip(start, data.shape))
        with h5py.File(self.path) as f:
            f[key][bb] = data


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
        with h5py.File(self.path) as f:
            dtype = f.attrs['dtype']

        if PipelineParameter().useN5Backend:
            labels = nz5.datasetWrapper(dtype, os.path.join(labelsPath, labelsKey))
        else:
            h5_file = nh5.openFile(labelsPath)
            labels = nh5.Hdf5Array(dtype, h5_file, labelsKey)
        nNodes = vigra.readHDF5(self.path, "numberOfNodes")
        return nrag.readStackedRagFromHdf5(labels, nNodes, self.path)

    # only read sub-parts
    def readKey(self, key):
        with h5py.File(self.path, 'r') as f:
            if key not in f.keys():
                print("The key", key, "is not in", f.keys())
                raise KeyError("Key not found!")
        return vigra.readHDF5(self.path, key)
