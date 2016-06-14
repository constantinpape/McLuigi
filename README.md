# LuigiCUt

Implementation of Multicut Workflow based on luigi.
This is mostly for deployment purposes. 
Still under development and quite unstable.

## Features

Implemented:

* Edge Features, Region Features, Topology Features
* RF Prediction
* MC Solver
* Blockwise MC Solver

Missing (and don't know what we actually want here yet):

* Generating Oversemgentation 
* RF Learning
* Lifted Multicut

## TODO next

* Consistent caching names (for now just the task name) -> understand caching better.
* Get central schedular running.
* Move parallelisation of blockwise stuff to scheduler, if possible.
* Backend for chunked h5.
* Chunked and blockwise processing.

## TODO then

* Data Backends: DVID, tiff ?
* Computational Backends: Spark, DVIDSparkServices
