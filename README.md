# McLuigi

Multicut workflow for large connectomics data.
Using luigi for pipelining and caching processing steps.
Most of the computations are done out-of-core using hdf5 as backend and implementations from nifty

## Installattion

To run the pipeline, you need the following python libraries:

* h5py
* Luigi: https://github.com/spotify/luigi (version ???)
* Vigra: https://github.com/ukoethe/vigra (master, needs to be build with hdf5 and python-bindings) 
* Nifty: https://github.com/constantinpape/nifty/tree/stacked_rag , needs to be build with python-bindings, hdf5, fastfilters (https://github.com/svenpeter42/fastfilters) and Gurobi or CPLEX.

The c++ libraries (vigra, nifty) need a compiler with c++ 11 support.

## Missing Features

* Generating oversemgentation (wsdt)
* Lifted Multicut
* Pipeline for isotropic data.

## TODOs

* Get central scheduler running.
* Doc + Examples
* Proper installer via setup.py/ conda
* Add docstrings
* change to PEP8 style
