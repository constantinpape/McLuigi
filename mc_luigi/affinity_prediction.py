from __future__ import print_function, division

import os
import argparse

from gunpowder import *
from gunpowder.caffe import *

def predict(
    raw_path,
    out_folder,
    out_name,
    net_architecture,
    net_weights,
    gpu_id
):
    assert os.path.exists(raw_path), raw_path
    assert os.path.exists(net_architecture), net_architecture
    assert os.path.exists(net_weights), net_weights

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # TODO this should not be hard coded here, should fetch it from somewhere instead
    input_size = Coordinate((84, 268, 268))
    output_size = Coordinate((56, 56, 56))

    # the size of the receptive field of the network
    context = (input_size - output_size) // 2

    # a chunk request that matches the dimensions of the network, will be used
    # to chunk the whole volume into batches of this size
    chunk_request = BatchRequest()
    chunk_request.add_volume_request(VolumeTypes.RAW, input_size)
    chunk_request.add_volume_request(VolumeTypes.PRED_AFFINITIES, output_size)

    #source = Hdf5Source(raw_path, datasets={VolumeTypes.RAW: 'volumes/raw'})
    source = Hdf5Source(raw_path, datasets={VolumeTypes.RAW: 'data'})

    # save the prediction
    save = Snapshot(
        every=1,
        output_dir=out_folder,
        output_filename=out_name
    )

    # build the pipeline
    pipeline = (

            # raw sources
            source +

            # normalize raw data
            Normalize() +

            # pad raw data
            Pad({ VolumeTypes.RAW: (100, 100, 100) }) +

            # shift to [-1, 1]
            IntensityScaleShift(2, -1) +

            ZeroOutConstSections() +

            # do the actual prediction
            Predict(net_architecture, net_weights, use_gpu=gpu_id) +

            PrintProfilingStats() +

            Chunk(chunk_request) +

            # save
            save
    )

    with build(pipeline) as p:

        # get the ROI of the whole RAW region from the source
        raw_roi = source.get_spec().volumes[VolumeTypes.RAW]

        # request affinity predictions for the whole RAW ROI
        whole_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.PRED_AFFINITIES: raw_roi.grow(-context, -context)
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        p.request_batch(whole_request)


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('net_architecture', type=str)
    parser.add_argument('net_weights', type=str)
    parser.add_argument('gpu_id', type=int)
    args = parser.parse_args()

    out_path = args.out_path
    out_folder, out_name = os.path.split(out_path)
    return args.raw_path, out_folder, out_name, args.net_architecture, args.net_weights, args.gpu_id


if __name__ == "__main__":
    predict(*parse_input())
