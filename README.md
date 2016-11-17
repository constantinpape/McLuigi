# McLuigi

Multicut workflow for large connectomics data.
Using luigi https://github.com/spotify/luigi for pipelining and caching processing steps.
Most of the computations are done out-of-core using hdf5 as backend and implementations from nifty
https://github.com/DerThorsten/nifty.

## Features

Implemented:

* Generating Oversemgentation  (wsdt)
* Edge Features, Region Features
* RF / XGB Learning
* RF / XGB Prediction
* MC Solver
* Blockwise MC Solver

Missing:

* Lifted Multicut
* Pipeline for isotropic data.

## TODOs

* Get central scheduler running.
* Doc + Examples
* Proper installer via setup.py/ conda
