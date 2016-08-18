import logging
import numpy as np

# call this function to configure the logger
# change the log file / log level if necessary
def config_logger(logger):

    # for now we have to change this here, until we have better configuration handling
    handler = logging.FileHandler('luigi_workflow.log')
    handler.setLevel(logging.INFO)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


#
# Implementation of a disjoint-set forest
# TODO maybe use C++ implementation instead
#

# Node datastrucuture for UDF
# works only for connected labels
class Node(object):
    def __init__(self, u):
        self.parent = self
        self.label  = u
        self.rank   = 0

# Uninion Find data structiure
class UnionFind(object):

    def __init__(self, n_labels):
        assert isinstance(n_labels, int), type(n_labels)
        self.n_labels = n_labels
        self.nodes = [Node(n) for n in xrange(n_labels)]


    # find the root of u and compress the path on the way
    def find(self, u_id):
        #assert u_id < self.n_labels
        u = self.nodes[ u_id ]
        return self.findNode(u)

    # find the root of u and compress the path on the way
    def findNode(self, u):
        if u.parent == u:
            return u
        else:
            u.parent = self.findNode(u.parent)
            return u.parent

    def merge(self, u_id, v_id):
        #assert u_id < self.n_labels
        #assert v_id < self.n_labels
        u = self.nodes[ u_id ]
        v = self.nodes[ v_id ]
        self.mergeNode(u, v)

    # merge u and v trees in a union by rank manner
    def mergeNode(self, u, v):
        u_root = self.findNode(u)
        v_root = self.findNode(v)
        if u_root.rank > v_root.rank:
            v_root.parent = u_root
        elif u_root.rank < v_root.rank:
            u_root.parent = v_root
        elif u_root != v_root:
            v_root.parent = u_root
            u_root.rank += 1

    # get the new sets after merging
    def get_merge_result(self):

        merge_result = []

        # find all the unique roots
        roots = []
        for u in self.nodes:
            root = self.findNode(u)
            if not root in roots:
                roots.append(root)

        # find ordering of roots (from 1 to n_roots)
        roots_ordered = {}
        root_id = 0
        for root in roots:
            merge_result.append( [] )
            roots_ordered[root] = root_id
            root_id += 1
        for u in self.nodes:
            u_label = u.label
            root = self.findNode(u)
            merge_result[ roots_ordered[root] ].append(u_label)

        # sort the nodes in the result
        #(this might result in problems if label_type cannot be sorted)
        for res in merge_result:
            res.sort()

        return merge_result


# aaaaah this is ugly
def get_blocks(shape, block_size, block_overlap):

    s_z = block_size[0]
    assert s_z < shape[0], str(s_z) + " , " + str(shape[0])
    s_y = block_size[1]
    assert s_y < shape[1], str(s_y) + " , " + str(shape[1])
    s_x = block_size[2]
    assert s_x < shape[2], str(s_x) + " , " + str(shape[2])

    o_z = block_overlap[0]
    o_y = block_overlap[1]
    o_x = block_overlap[2]

    n_z = int( np.ceil( float( shape[0] ) / s_z ) )
    n_y = int( np.ceil( float( shape[1] ) / s_y ) )
    n_x = int( np.ceil( float( shape[2] ) / s_x ) )

    n_blocks = n_x * n_y * n_z

    workflow_logger.info("Fitting " + str(n_blocks) + " blocks of size " + str(block_size) + " into shape " + str(shape) + " additional overlaps: " + str(block_overlap))

    block_begins = []
    block_ends   = []
    for z in xrange(n_z):

        # z range
        start_z = z * s_z
        if z != 0:
            start_z -= o_z
        end_z = (z + 1) * s_z + o_z
        if end_z > shape[0]:
            end_z = shape[0]

        for y in xrange(n_y):

            # Y range
            start_y = y * s_y
            if y != 0:
                start_y -= o_y
            end_y = (y + 1) * s_y + o_y
            if end_y > shape[1]:
                end_y = shape[1]

            for x in xrange(n_x):

                # x range
                start_x = x * s_x
                if x != 0:
                    start_x -= o_x
                end_x = (x + 1) * s_x + o_x
                if end_x > shape[2]:
                    end_x = shape[2]

                block_begins.append( [start_z,start_y,start_x] )
                block_ends.append(   [end_z,end_y,end_x] )

    return n_blocks, block_begins, block_ends
