import vigra

from cremi.evaluation import NeuronIds
from cremi import Volume


def evaluate(gt, segmentation):
    gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=1)
    evaluate = NeuronIds(Volume(gt))

    segmentation = Volume(segmentation)
    vi_split, vi_merge = evaluate.voi(segmentation)
    ri = evaluate.adapted_rand(segmentation)

    return vi_split, vi_merge, ri


def regression_test(
        ref_seg,
        seg,
        expected_vi_split=0,
        expected_vi_merge=0,
        expected_ri=0
):
    vi_split, vi_merge, ri = evaluate(ref_seg, seg)
    vi_s_pass = vi_split < expected_vi_split
    vi_m_pass = vi_merge < expected_vi_merge
    ri_pass   = ri < expected_ri
    if vi_m_pass and vi_s_pass and ri_pass:
        print "All passed with:"
        print "Vi-Split:", vi_split, "(Ref:)", expected_vi_split
        print "Vi-Merge:", vi_merge, "(Ref:)", expected_vi_merge
        print "RI:", ri, "(Ref:)", expected_ri
    else:
        print "FAILED with"
        print "Vi-Split: %s with %f, (Ref:) %f" % ('Passed' if vi_s_pass else 'Failed', vi_split, expected_vi_split)
        print "Vi-Merge: %s with %f, (Ref:) %f" % ('Passed' if vi_m_pass else 'Failed', vi_merge, expected_vi_merge)
        print "RI: %s with %f, (Ref:) %f" % ('Passed' if ri_pass else 'Failed', ri, expected_ri)


def clean_up():
    pass
