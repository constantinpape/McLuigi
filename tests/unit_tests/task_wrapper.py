# hack to run multiple luigi tasks in one session
import luigi
import json
import argparse

from mc_luigi import LearnClassifierFromGt, PipelineParameter
from mc_luigi import MulticutSegmentation, BlockwiseMulticutSegmentation


def learn_rf():
    ppl_parameter = PipelineParameter()
    ppl_parameter.read_input_file('./inputs.json')
    inputs = ppl_parameter.inputs
    ppl_parameter.separateEdgeClassification = True

    luigi.run([
        "--local-scheduler",
        "--pathsToSeg", json.dumps(inputs["seg"]),
        "--pathsToGt", json.dumps(inputs["gt"])],
        LearnClassifierFromGt
    )


def mc():
    ppl_parameter = PipelineParameter()
    # read the json with paths to input files
    ppl_parameter.read_input_file('./inputs.json')
    inputs = ppl_parameter.inputs

    # set the parameters
    ppl_parameter.multicutWeightingScheme = "z"
    ppl_parameter.separateEdgeClassification = True
    ppl_parameter.multicutVerbose = True

    print(inputs['rf'])
    luigi.run(
        ["--local-scheduler",
         "--pathToSeg", inputs["seg"][0],
         "--pathToClassifier", inputs["rf"]],
        MulticutSegmentation
    )


def blockwise_mc():
    ppl_parameter = PipelineParameter()
    # read the json with paths to input files
    ppl_parameter.read_input_file('./inputs.json')
    inputs = ppl_parameter.inputs

    # set the parameters
    ppl_parameter.multicutWeightingScheme = "z"
    ppl_parameter.separateEdgeClassification = True
    ppl_parameter.multicutVerbose = True
    ppl_parameter.multicutBlockShape = [50, 512, 512]
    ppl_parameter.multicutBlockOverlap = [5, 50, 50]

    n_levels = 1
    luigi.run(
        ["--local-scheduler",
         "--pathToSeg", inputs["seg"][0],
         "--pathToClassifier", inputs["rf"],
         "--numberOfLevels", str(n_levels)],
        BlockwiseMulticutSegmentation
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    args = parser.parse_args()
    task_fu = eval(args.task)
    task_fu()


if __name__ == '__main__':
    main()
