# McLuigi

Multicut workflow for large connectomics data.
Using luigi for pipelining and caching processing steps.
Most of the computations are done out-of-core using hdf5 as backend and implementations from nifty

## Installation

You can either build the following dependencies from source:

* Luigi: https://github.com/spotify/luigi (version 2.3)
* Vigra: https://github.com/ukoethe/vigra (master, needs to be build with hdf5 and python-bindings) 
* Nifty: https://github.com/constantinpape/nifty/tree/stacked_rag , needs to be build with python-bindings, hdf5, fastfilters (https://github.com/svenpeter42/fastfilters) and Gurobi or CPLEX.

or (recommended) install via conda.

```
$ conda install -c cpape mc_luigi=0.1 
```

To fulfill all dependencies, you also need to have the ilastik channel in your .condarc .

## Missing Features

* Lifted Multicut
* Pipeline for isotropic data.

## TODOs

* Get central scheduler running.
* Add docstrings
* change to PEP8 style
