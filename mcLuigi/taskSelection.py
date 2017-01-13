# Multicut Pipeline implemented with luigi
# Selecting appropriate tasks for several high-level
# pipeline options

from featureTasks import EdgeFeatures, RegionFeatures
from defectHandlingTasks import ModifiedEdgeFeatures, ModifiedRegionFeatures, ModifiedMulticutProblem
from multicutProblemTasks import StandardMulticutProblem

from pipelineParameter import PipelineParameter
from tools import config_logger

import logging

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


# TODO select debug tasks if flag
# try vigra features for debugging
#from debugTasks import get_local_vigra_features
#get_local_features = get_local_vigra_features

# TODO implement and select isotropic tasks if not pipelineParams.anisotropicPipeline

# select defect handling tasks if pipelineParams.defectPipeline
if PipelineParameter().defectPipeline:
    MulticutProblem = ModifiedMulticutProblem
else:
    MulticutProblem = StandardMulticutProblem


# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features(xyOnly = False, zOnly = False):

    assert not (xyOnly and zOnly)

    # load the paths to input files
    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)
    # load the feature config
    with open(PipelineParameter().FeatureConfigFile, 'r') as f:
        feat_params = json.load(f)

    # choose the appropriate feature tasks (normal vs. defect handling) <- pipelineParams.defectPipeline
    # TODO (anisotropic vs. isotropic) <- pipelineParams.anisotropicPipeline
    if PipelineParameter().defectPipeline:
        workflow_logger.debug("Using tasks modified for defect handling in get_local_features")
        EdgeTask   = ModifiedEdgeFeatures
        RegionTask = ModifiedRegionFeatures
    else:
        workflow_logger.debug("Using standard tasks in get_local_features")
        EdgeTask   = EdgeFeatures
        RegionTask = RegionFeatures

    features = feat_params["features"]

    if not isinstance(features, list):
        features = [features,]

    feature_tasks = []

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data,]

    if "raw" in features:
        # by convention we assume that the raw data is given as 0th
        feature_tasks.append( EdgeTask(input_data[0], inputs["seg"]) )
        workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[0])

    if "prob" in features:
        # by convention we assume that the membrane probs are given as 1st
        feature_tasks.append( EdgeTask(input_data[1], inputs["seg"] ) )
        workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[1])

    if "affinitiesXY" in features and not zOnly: # specific XY - features -> we keep only these
        # by convention we assume that the xy - affinity channel is given as 1st input
        feature_tasks.append( EdgeTask(input_data[1], inputs["seg"], keepOnlyXY = True ) )
        workflow_logger.debug("Calculating Edge Features from xy affinity maps: " + input_data[1])

    if "affinitiesZ" in features and not xyOnly: # specific Z - features -> we keep only these
        # by convention we assume that the z - affinity channel is given as 2nd input
        feature_tasks.append( EdgeTask(input_data[2], inputs["seg"], keepOnlyZ = True ) )
        workflow_logger.debug("Calculating Edge Features from z affinity maps: " + input_data[2])

    if "reg" in features:
        # by convention we calculate region features only on the raw data (0th input)
        feature_tasks.append( RegionTask(input_data[0], inputs["seg"]) )
        workflow_logger.debug("Calculating Region Features")

    # check for invalid keys
    for kk in features:
        if kk not in ("raw", "prob", "affinitiesXY", "affinitiesZ", "reg"):
            workflow_logger.info("get_local_features: Ignoring unknown key %s" % kk)

    #if "topo" in features:
    #    # by convention we calculate region features only on the raw data (0th input)
    #    feature_tasks.append( TopologyFeatures(inputs["seg"], features2d ) )
    #    workflow_logger.debug("Calculating Topology Features")

    return feature_tasks


# read the feature configuration from PipelineParams.FeatureConfigFile
# and return the corresponding feature tasks
def get_local_features_for_multiinp(xyOnly = False, zOnly = False):

    assert not (xyOnly and zOnly)

    # load the paths to input files
    with open(PipelineParameter().InputFile, 'r') as f:
        inputs = json.load(f)
    # load the feature config
    with open(PipelineParameter().FeatureConfigFile, 'r') as f:
        feat_params = json.load(f)

    # choose the appropriate feature tasks (normal vs. defect handling) <- pipelineParams.defectPipeline
    # TODO (anisotropic vs. isotropic) <- pipelineParams.anisotropicPipeline
    if PipelineParameter().defectPipeline:
        workflow_logger.debug("Using tasks modified for defect handling in get_local_features_for_multiinp")
        EdgeTask   = ModifiedEdgeFeatures
        RegionTask = ModifiedRegionFeatures
    else:
        workflow_logger.debug("Using standard tasks in get_local_features_for_multiinp")
        EdgeTask   = EdgeFeatures
        RegionTask = RegionFeatures

    features = feat_params["features"]
    if not isinstance(features, list):
        features = [features,]

    input_data = inputs["data"]
    if not isinstance(input_data, list):
        input_data = [input_data,]

    segs = inputs["seg"]

    nInpPerSeg = len(input_data) / len(segs)

    feature_tasks = []
    for i in xrange(len(segs)):
        inp0 = nInpPerSeg*i
        inp1 = nInpPerSeg*i + 1
        inp2 = nInpPerSeg*i + 2

        feature_tasks.append([])

        if "raw" in features:
            # by convention we assume that the raw data is given as 0th
            feature_tasks[i].append( EdgeTask(input_data[inp0], segs[i]) )
            workflow_logger.debug("Calculating Edge Features from raw input: " + input_data[inp0])

        if "prob" in features:
            #assert nInpPerSeg == 2
            # by convention we assume that the membrane probs are given as 1st
            feature_tasks[i].append( EdgeTask(input_data[inp1], segs[i] ) )
            workflow_logger.debug("Calculating Edge Features from probability maps: " + input_data[inp1])

        if "affinitiesXY" in features and not zOnly:
            assert nInpPerSeg == 3
            # by convention we assume that the xy - affinity channel is given as 1st input
            feature_tasks[i].append( EdgeTask(input_data[inp1], segs[i], keepOnlyXY = True ) )
            workflow_logger.debug("Calculating Edge Features from xy affinity maps: " + input_data[inp1])

        if "affinitiesZ" in features and not xyOnly:
            assert nInpPerSeg == 3
            # by convention we assume that the z - affinity channel is given as 2nd input
            feature_tasks[i].append( EdgeTask(input_data[inp2], segs[i], keepOnlyZ = True ) )
            workflow_logger.debug("Calculating Edge Features from z affinity maps: " + input_data[inp2])

        if "reg" in features:
            feature_tasks[i].append( RegionTask(input_data[inp0], segs[i]) )
            workflow_logger.debug("Calculating Region Features")

    # check for invalid keys
    for kk in features:
        if kk not in ("raw", "prob", "affinitiesXY", "affinitiesZ", "reg"):
            workflow_logger.info("get_local_features: Ignoring unknown key %s" % kk)

    return feature_tasks
