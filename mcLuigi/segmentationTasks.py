import luigi
import numpy as np
import time
import logging
import os
from concurrent import futures

# TODO Implement 2d/3d version in vigra and use vigra function instead to get rid of
# dependency on wsdt
#from vigra.analysis import wsDtSegmentation as compute_wsdt_segmentation
from wsdt import wsDtSegmentation as compute_wsdt_segmentation

from dataTasks import InputData
from customTargets import HDF5VolumeTarget

from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)

class WsdtSegmentation(luigi.Task):
    """
    Task for generating segmentation via wsdt.
    """

    pathToProbabilities = luigi.Parameter()
    dtype = luigi.Parameter(default = 'uint32')
    # TODO get these from ppl params
    #WatershedParameter = luigi.DictParameter()

    def requires(self):
        """
        Dependencies:
        """
        return InputData(self.pathToProbabilities)

    # TODO enable 3d wsdt for isotropic ppl
    # TODO we could integrate the (statistics based) defect detection directly in here
    # TODO parallelisation is really weird / slow, investigate !
    # -> check if same strange behaviour on sirherny
    # if not, probably nifty-package is responsible -> hdf5 parallel read/write ?! (libhdf5-serial ?)
    # -> if this is true, also other applications should be horribly slow
    @run_decorator
    def run(self):
        pmap = self.input()
        pmap.open()
        shape = pmap.shape()
        out  = self.output()
        out.open(shape, pmap.chunk_shape() )

        self._run_wsdt2d_standard(pmap, out, shape)

        out.close()
        pmap.close()


    def _run_wsdt2d_standard(self, pmap, out, shape):

        # read the wsdt settings from ppl params
        ppl_params = PipelineParameter()
        threshold   = ppl_params.wsdtThreshold
        min_mem     = ppl_params.wsdtMinMem
        min_seg     = ppl_params.wsdtMinSeg
        sig_seeds   = ppl_params.wsdtSigSeeds
        sig_weights = ppl_params.wsdtSigWeights
        invert      = ppl_params.wsdtInvert
        workflow_logger.info(
                "WsdtSegmentation: Running standard 2d dt watershed with threshold %f" % threshold)

        def segment_slice(z):
            print "Slice", z, "/", shape[0]
            sliceStart = [z,0,0]
            sliceStop  = [z+1,shape[1],shape[2]]
            # TODO proper params from ppl. params
            pmap_z = pmap.read(sliceStart, sliceStop)
            if invert:
                pmap_z = 1. - pmap_z
            seg, max_z = compute_wsdt_segmentation(pmap_z.squeeze(),
                    threshold,
                    min_mem, min_seg,
                    sig_seeds, sig_weights,
                    groupSeeds = False) # group seeds set to false for proper threading behaviour (everything should have gil lifted)
            if max_z == 0: # this can happen for defected slices
                max_z = 1
            else:
                # default minval of wsdt segmentation is 1
                seg -= 1
            #if not 0 in seg:
            #    seg,_,_ = vigra.analysis.relabelConsecutive(seg, start_label = 0, keep_zeros = False)
            out.write(sliceStart, seg[None,:,:])
            return max_z

        t_ws = time.time()
        # sequential for debugging
        #offsets = []
        #for z in xrange(shape[0]):
        #    offsets.append( segment_slice(z) )

        # parallel
        nWorkers = PipelineParameter().nThreads
        ##nWorkers = 20
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for z in xrange(shape[0]):
               tasks.append(executor.submit(segment_slice,z))
            offsets = [future.result() for future in tasks]
        workflow_logger.info("WsdtSegmentation: Running watershed took: %f s" % (time.time() - t_ws) )

        # accumulate the offsets for each slice
        offsets = np.roll(offsets, 1)
        offsets[0] = 0
        offsets = np.cumsum(offsets)

        if offsets[-1] >= 4294967295 and self.dtype == np.dtype('uint32'):
            print "WARNING: UINT32 overflow!"
            # we don't add the offset, but only save the non-offset file
            return True

        def add_offset(z, offset):
            #print z, '/', shape[0]
            sliceStart = [z,0,0]
            sliceStop  = [z+1,shape[1],shape[2]]
            seg = out.read(sliceStart, sliceStop)
            seg += offset
            out.write(sliceStart, seg)
            return True

        # sequential
        #for z in xrange(shape[0]):
        #    add_offset(z, offsets[z])

        t_off = time.time()
        # parallel
        with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            tasks = []
            for z in xrange(shape[0]):
                tasks.append( executor.submit(add_offset, z, offsets[z]) )
            res = [t.result() for t in tasks]
        workflow_logger.info("WsdtSegmentation: Adding offsets took %f s" % (time.time() - t_off) )


    # TODO if we integrate defects / 3d ws reflact this in the caching name
    def output(self):
        save_path = os.path.join( PipelineParameter().cache,
                "WsdtSegmentation_%s.h5" % os.path.split(self.pathToProbabilities)[1][:-3] )
        return HDF5VolumeTarget( save_path, self.dtype, compression = PipelineParameter().compressionLevel )
