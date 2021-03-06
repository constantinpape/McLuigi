from __future__ import division, print_function

# Multicut Pipeline implemented with luigi
# Selecting appropriate feature tasks for several high-level pipeline options

from .featureTasks import EdgeFeatures, RegionFeatures
from .pipelineParameter import PipelineParameter
from .tools import config_logger

import logging

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# TODO TODO change the whole thing to dictionary based json inputs !!!!
# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features():

    # load the paths to input files
    inputs = PipelineParameter().inputs
    keys = PipelineParameter().input_keys

    # TODO (anisotropic vs. isotropic) <- pipelineParams.anisotropicPipeline -> also move to feature tasks!

    features = PipelineParameter().features
    if not isinstance(features, list):
        features = [features]
    feature_tasks = []

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data]
    input_keys = None if keys is None else keys['data']

    seg = inputs["seg"]
    if isinstance(seg, list):
        assert len(seg) == 1, str(seg)
        seg = seg[0]

    simpleFeats = PipelineParameter().useSimpleFeatures
    zDir = PipelineParameter().zAffinityDirection

    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append(EdgeFeatures(pathToInput=input_data[0],
                                          pathToSeg=seg,
                                          simpleFeatures=simpleFeats,
                                          keyToSeg='data' if keys is None else keys['seg'],
                                          keyToInput='data' if input_keys is None else input_keys[0]))
        workflow_logger.debug("get_local_features: calculating Edge Features from raw input: " + input_data[0])

    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append(EdgeFeatures(pathToInput=input_data[1],
                                          pathToSeg=seg,
                                          simpleFeatures=simpleFeats,
                                          keyToSeg='data' if keys is None else keys['seg'],
                                          keyToInput='data' if input_keys is None else input_keys[1]))
        workflow_logger.debug("get_local_features: calculating Edge Features from probability maps: " + input_data[1])

    if "affinitiesXY" in features:  # specific XY - features -> we keep only these
        # by convention we assume that the xy - affinity channel is given as 1st input
        feature_tasks.append(EdgeFeatures(pathToInput=input_data[1],
                                          pathToSeg=seg,
                                          keepOnlyXY=True,
                                          simpleFeatures=simpleFeats,
                                          keyToSeg='data' if keys is None else keys['seg'],
                                          keyToInput='data' if input_keys is None else input_keys[1]))
        workflow_logger.debug("get_local_features: calculating Edge Features from xy affinity maps: " + input_data[1])

    if "affinitiesZ" in features:  # specific Z - features -> we keep only these
        # by convention we assume that the z - affinity channel is given as 2nd input
        feature_tasks.append(EdgeFeatures(pathToInput=input_data[2],
                                          pathToSeg=seg,
                                          keepOnlyZ=True,
                                          zDirection=zDir,
                                          simpleFeatures=simpleFeats,
                                          keyToSeg='data' if keys is None else keys['seg'],
                                          keyToInput='data' if input_keys is None else input_keys[2]))
        workflow_logger.debug("get_local_features: calculating Edge Features from z affinity maps: " + input_data[2])

    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        feature_tasks.append(RegionFeatures(pathToInput=input_data[0],
                                            pathToSeg=seg,
                                            keyToSeg='data' if keys is None else keys['seg'],
                                            keyToInput='data' if input_keys is None else input_keys[0]))
        workflow_logger.debug("get_local_features: calculating Region Features")

    # check for invalid keys
    for kk in features:
        if kk not in ("raw", "prob", "affinitiesXY", "affinitiesZ", "reg"):
            workflow_logger.info("get_local_features: Ignoring unknown key %s" % kk)

    return feature_tasks


# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features_for_multiinp():

    # load the paths to input files
    inputs = PipelineParameter().inputs

    # choose the appropriate feature tasks (normal vs. defect handling) <- pipelineParams.defectPipeline
    # TODO (anisotropic vs. isotropic) <- pipelineParams.anisotropicPipeline
    features = PipelineParameter().features
    if not isinstance(features, list):
        features = [features]

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data]

    segs = inputs["seg"]
    nInpPerSeg = len(input_data) / len(segs)

    simpleFeats = PipelineParameter().useSimpleFeatures
    zDir = PipelineParameter().zAffinityDirection

    feature_tasks = []
    for i in range(len(segs)):
        inp0 = int(nInpPerSeg * i)
        inp1 = int(nInpPerSeg * i + 1)
        inp2 = int(nInpPerSeg * i + 2)

        feature_tasks.append([])

        if "raw" in features:
            # by convention we assume that the raw data is given as 0th
            feature_tasks[i].append(EdgeFeatures(input_data[inp0], segs[i], simpleFeatures=simpleFeats))
            workflow_logger.debug(
                "get_local_features_for_multiinp: calculating Edge Features from raw input: " + input_data[inp0]
            )

        if "prob" in features:
            # by convention we assume that the membrane probs are given as 1st
            feature_tasks[i].append(EdgeFeatures(input_data[inp1], segs[i], simpleFeatures=simpleFeats))
            workflow_logger.debug(
                "get_local_features_for_multiinp: calculating Edge Features from probability maps: " + input_data[inp1]
            )

        if "affinitiesXY" in features:
            assert nInpPerSeg == 3
            # by convention we assume that the xy - affinity channel is given as 1st input
            feature_tasks[i].append(EdgeFeatures(input_data[inp1],
                                                 segs[i],
                                                 keepOnlyXY=True,
                                                 simpleFeatures=simpleFeats))
            workflow_logger.debug(
                "get_local_features_for_multiinp: calculating Edge Features from xy affinity maps: " + input_data[inp1]
            )

        if "affinitiesZ" in features:
            assert nInpPerSeg == 3
            # by convention we assume that the z - affinity channel is given as 2nd input
            feature_tasks[i].append(EdgeFeatures(input_data[inp2],
                                                 segs[i],
                                                 keepOnlyZ=True,
                                                 zDirection=zDir,
                                                 simpleFeatures=simpleFeats))
            workflow_logger.debug(
                "get_local_features_for_multiinp: calculating Edge Features from z affinity maps: " + input_data[inp2]
            )

        if "reg" in features:
            feature_tasks[i].append(RegionFeatures(input_data[inp0], segs[i]))
            workflow_logger.debug("get_local_features_for_multiinp: calculating Region Features")

    # check for invalid keys
    for kk in features:
        if kk not in ("raw", "prob", "affinitiesXY", "affinitiesZ", "reg"):
            workflow_logger.info("get_local_features_for_multiinp: Ignoring unknown key %s" % kk)

    return feature_tasks
