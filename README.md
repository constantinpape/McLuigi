# McLuigi

Multicut workflow for large connectomics data.
Using luigi for pipelining and caching processing steps.
Most of the computations are done out-of-core using hdf5 as backend and implementations from nifty

## Installation

You need a cplex or gurobi license to use the multicut pipeline.

### conda (recommended)

To fulfill all dependencies, you also need to have the ilastik channel in your .condarc .
Then install the package via
```
$ GUROBI_ROOT_DIR=/path/to/gurobi conda install -c cpape mc_luigi=0.1 
```
The current package only supports gurobi as solver backend, please make an issue if you need a version with cplex.

### From source

You can also install the dependencies from source:

* Luigi: https://github.com/spotify/luigi (version 2.3)
* Vigra: https://github.com/ukoethe/vigra (master, needs to be build with hdf5 and python-bindings)
* Nifty: https://github.com/constantinpape/nifty/tree/stacked_rag , needs to be build with python-bindings, hdf5, fastfilters (https://github.com/svenpeter42/fastfilters) and Gurobi or CPLEX.


## Missing Features

* Lifted Multicut
* Pipeline for isotropic data.

## TODOs

* Get central scheduler running.
* Add docstrings
* change to PEP8 style
